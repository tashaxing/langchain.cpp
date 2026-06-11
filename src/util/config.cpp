// langchain/util/config.cpp

#include "util/config.h"

#include <algorithm>
#include <cstdlib>
#include <regex>
#include <stdexcept>

namespace langchain
{
namespace util
{

namespace
{

// Interpolate ${VAR} or ${VAR:-default} from environment variables.
std::string interpolate_env(const std::string& value)
{
    static const std::regex re(R"(\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\})");
    std::string result = value;
    std::smatch match;

    // Keep searching from the start because replacements may contain '$'.
    std::size_t search_pos = 0;
    while (search_pos < result.size())
    {
        std::string sub = result.substr(search_pos);
        if (!std::regex_search(sub, match, re))
        {
            break;
        }

        std::size_t match_pos = search_pos + match.position();
        std::string var_name = match[1].str();
        std::string default_val = match[2].str();

        const char* env_val = std::getenv(var_name.c_str());
        std::string replacement = (env_val && *env_val) ? env_val : default_val;

        result.replace(match_pos, match.length(), replacement);
        search_pos = match_pos + replacement.size();
    }

    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool Config::load(const std::string& path)
{
    std::filesystem::path fs_path(path);
    std::unique_lock lock(mutex_);
    if (!load_unlocked(fs_path))
        return false;
    file_path_ = std::move(fs_path);
    return true;
}

bool Config::save()
{
    std::unique_lock lock(mutex_);
    if (file_path_.empty())
        return false;
    return save_unlocked(file_path_);
}

bool Config::save_as(const std::string& path)
{
    std::filesystem::path fs_path(path);
    std::unique_lock lock(mutex_);
    if (!save_unlocked(fs_path))
        return false;
    file_path_ = std::move(fs_path);
    return true;
}

bool Config::check_reload()
{
    std::unique_lock lock(mutex_);
    if (file_path_.empty())
        return false;

    std::error_code ec;
    auto current = std::filesystem::last_write_time(file_path_, ec);
    if (ec)
        return false;

    if (current == last_write_time_)
        return false;

    return load_unlocked(file_path_);
}

bool Config::reload()
{
    std::unique_lock lock(mutex_);
    if (file_path_.empty())
        return false;
    return load_unlocked(file_path_);
}

ConfigSection& Config::section(const std::string& name)
{
    std::unique_lock lock(mutex_);
    return sections_[name];
}

const ConfigSection& Config::section(const std::string& name) const
{
    std::shared_lock lock(mutex_);
    auto it = sections_.find(name);
    if (it == sections_.end())
        throw std::out_of_range("ConfigSection not found: " + name);
    return it->second;
}

bool Config::has_section(const std::string& name) const
{
    std::shared_lock lock(mutex_);
    return sections_.find(name) != sections_.end();
}

void Config::remove_section(const std::string& name)
{
    std::unique_lock lock(mutex_);
    sections_.erase(name);
}

std::vector<std::string> Config::section_names() const
{
    std::shared_lock lock(mutex_);
    std::vector<std::string> names;
    names.reserve(sections_.size());
    for (const auto& [name, _] : sections_)
        names.push_back(name);
    return names;
}

std::string Config::file_path() const
{
    std::shared_lock lock(mutex_);
    return file_path_.string();
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
std::vector<std::string> Config::validate(const std::vector<Rule>& rules) const
{
    std::shared_lock lock(mutex_);
    std::vector<std::string> errors;

    for (const auto& rule : rules)
    {
        auto sec_it = sections_.find(rule.section);
        if (sec_it == sections_.end())
        {
            if (rule.required)
            {
                errors.push_back("missing required section: '" + rule.section + "'");
            }
            continue;
        }

        if (!sec_it->second.has(rule.key))
        {
            if (rule.required)
            {
                errors.push_back("missing required key '" + rule.key +
                                 "' in section '" + rule.section + "'");
            }
            continue;
        }

        if (rule.check)
        {
            std::string value = sec_it->second.get(rule.key, std::string{});
            if (!rule.check(value))
            {
                errors.push_back("invalid value for key '" + rule.key +
                                 "' in section '" + rule.section + "': '" +
                                 value + "'");
            }
        }
    }

    return errors;
}

// ---------------------------------------------------------------------------
// Internal: load XML without locking.
// ---------------------------------------------------------------------------
bool Config::load_unlocked(const std::filesystem::path& path)
{
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());
    if (!result)
        return false;

    sections_.clear();

    pugi::xml_node root = doc.child("config");
    if (!root)
        root = doc.first_child(); // fallback: accept any root

    for (pugi::xml_node sec_node : root.children())
    {
        std::string sec_name = sec_node.name();
        if (sec_name.empty())
            continue;

        ConfigSection sec(sec_name);

        for (pugi::xml_node kv : sec_node.children())
        {
            std::string key = kv.name();
            std::string value = kv.child_value();
            sec.set(key, interpolate_env(value));
        }
        sections_[std::move(sec_name)] = std::move(sec);
    }

    std::error_code ec;
    last_write_time_ = std::filesystem::last_write_time(path, ec);
    return true;
}

// ---------------------------------------------------------------------------
// Internal: save XML without locking.
// ---------------------------------------------------------------------------
bool Config::save_unlocked(const std::filesystem::path& path)
{
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("config");

    for (const auto& [name, sec] : sections_)
    {
        pugi::xml_node sec_node = root.append_child(name.c_str());

        for (const auto& [key, value] : sec.snapshot())
        {
            pugi::xml_node kv = sec_node.append_child(key.c_str());
            kv.append_child(pugi::node_pcdata).set_value(value.c_str());
        }
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        return false;

    doc.save(ofs, "  ");
    ofs.close();

    std::error_code ec;
    last_write_time_ = std::filesystem::last_write_time(path, ec);
    return true;
}

} // namespace util
} // namespace langchain
