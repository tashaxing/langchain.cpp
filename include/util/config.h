// langchain/util/config.h
// Hot-reloadable XML configuration manager.
//
// Sections group related key/value pairs.  The Config singleton owns all
// sections and watches the backing file for changes.
//
// Usage:
//   auto& cfg = langchain::util::Singleton<langchain::util::Config>::instance();
//   cfg.load("app.xml");
//   auto& db = cfg.section("database");
//   std::string host = db.get<std::string>("host", "localhost");
//   int port         = db.get<int>("port", 3306);
//
//   db.set("port", 3307);          // runtime mutation
//   cfg.save();                     // persist back to disk
//
//   if (cfg.check_reload()) {       // hot-reload when file changed
//       // sections were refreshed from disk
//   }
//
// XML layout (see config/app_config_template.xml):
//   <config>
//     <database>
//       <enabled>true</enabled>
//       <host>127.0.0.1</host>
//       <port>3306</port>
//     </database>
//   </config>
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pugixml.hpp"
#include "singleton.h"

namespace langchain
{
namespace util
{

// ---------------------------------------------------------------------------
// ConfigSection — a named group of string key/value pairs.
// ---------------------------------------------------------------------------
class ConfigSection
{
public:
    explicit ConfigSection(std::string name = {}) : name_(std::move(name)) {}

    const std::string& name() const { return name_; }

    // Type-safe getter with default fallback.
    template <typename T>
    T get(const std::string& key, const T& default_value) const
    {
        std::shared_lock lock(mutex_);
        auto it = values_.find(key);
        if (it == values_.end())
            return default_value;
        return from_string<T>(it->second);
    }

    // std::string overload avoids extra copy/conversion.
    std::string get(const std::string& key, const std::string& default_value) const
    {
        std::shared_lock lock(mutex_);
        auto it = values_.find(key);
        return (it == values_.end()) ? default_value : it->second;
    }

    // Type-safe setter.
    template <typename T>
    void set(const std::string& key, const T& value)
    {
        std::unique_lock lock(mutex_);
        values_[key] = to_string(value);
    }

    // Raw string setter (avoids template instantiation for strings).
    void set(const std::string& key, std::string value)
    {
        std::unique_lock lock(mutex_);
        values_[key] = std::move(value);
    }

    bool has(const std::string& key) const
    {
        std::shared_lock lock(mutex_);
        return values_.find(key) != values_.end();
    }

    void remove(const std::string& key)
    {
        std::unique_lock lock(mutex_);
        values_.erase(key);
    }

    std::unordered_map<std::string, std::string> snapshot() const
    {
        std::shared_lock lock(mutex_);
        return values_;
    }

    void clear()
    {
        std::unique_lock lock(mutex_);
        values_.clear();
    }

private:
    std::string name_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::string> values_;

public:
    ConfigSection(const ConfigSection& other)
    {
        std::shared_lock lock(other.mutex_);
        name_ = other.name_;
        values_ = other.values_;
    }

    ConfigSection& operator=(const ConfigSection& other)
    {
        if (this != &other) {
            std::unique_lock lock(mutex_);
            std::shared_lock lock_other(other.mutex_);
            name_ = other.name_;
            values_ = other.values_;
        }
        return *this;
    }

    ConfigSection(ConfigSection&& other) noexcept
    {
        std::unique_lock lock(other.mutex_);
        name_ = std::move(other.name_);
        values_ = std::move(other.values_);
    }

    ConfigSection& operator=(ConfigSection&& other) noexcept
    {
        if (this != &other) {
            std::unique_lock lock(mutex_);
            std::unique_lock lock_other(other.mutex_);
            name_ = std::move(other.name_);
            values_ = std::move(other.values_);
        }
        return *this;
    }

    // ---- conversion helpers ----
    template <typename T>
    static std::string to_string(const T& v)
    {
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }

    static std::string to_string(const std::string& s) { return s; }
    static std::string to_string(std::string&& s) { return std::move(s); }

    template <typename T>
    static T from_string(const std::string& s)
    {
        std::istringstream iss(s);
        T v{};
        iss >> v;
        return v;
    }

    static const std::string& from_string(const std::string& s) { return s; }
};

// ---------------------------------------------------------------------------
// Config — singleton configuration manager backed by an XML file.
// ---------------------------------------------------------------------------
class Config
{
public:
    // Load configuration from an XML file.  Returns false on I/O or parse error.
    bool load(const std::string& path);

    // Save current in-memory configuration back to the loaded file path.
    // Returns false on I/O error.
    bool save();

    // Save to an explicit path (also updates the watched path).
    bool save_as(const std::string& path);

    // Check whether the backing file has changed on disk; if so, reload it.
    // Returns true when a reload actually happened.
    bool check_reload();

    // Force a reload from the currently-set file path.
    bool reload();

    // Access a section by name.  Creates an empty section if it does not exist.
    ConfigSection& section(const std::string& name);

    // Const access; throws std::out_of_range if missing.
    const ConfigSection& section(const std::string& name) const;

    bool has_section(const std::string& name) const;

    void remove_section(const std::string& name);

    std::vector<std::string> section_names() const;

    // Path currently being watched / persisted to.
    std::string file_path() const;

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------
    // A rule checks that a key exists and optionally validates its value.
    struct Rule
    {
        std::string section;
        std::string key;
        bool required = true;          // if true, key must exist
        std::function<bool(const std::string&)> check; // optional value validator
    };

    // Validate configuration against a set of rules.
    // Returns empty vector on success, or a list of error messages.
    std::vector<std::string> validate(const std::vector<Rule>& rules) const;

private:
    friend class Singleton<Config>;
    Config() = default;

    mutable std::shared_mutex mutex_;
    std::filesystem::path file_path_;
    std::filesystem::file_time_type last_write_time_;
    std::unordered_map<std::string, ConfigSection> sections_;

    bool load_unlocked(const std::filesystem::path& path);
    bool save_unlocked(const std::filesystem::path& path);
};

} // namespace util
} // namespace langchain
