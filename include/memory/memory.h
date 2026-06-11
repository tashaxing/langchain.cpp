// langchain/memory/memory.h
// Conversation memory abstractions: short-term (in-memory) and long-term
// (persistent) implementations.
//
// Short-term:
//   BufferMemory    -- keeps every message in memory.
//   WindowMemory    -- keeps only the last k exchanges (sliding window).
//
// Long-term (persistent):
//   LongTermMemory  -- unified interface with selectable backend (JSON file
//                      or SQLite). Use LongTermMemory::json_file() or
//                      LongTermMemory::sqlite() for convenience.
#pragma once

#include "util/common.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace langchain
{
namespace memory
{

// ---------------------------------------------------------------------------
// IMemory -- base interface shared by all memory implementations.
// ---------------------------------------------------------------------------
class IMemory
{
public:
    virtual ~IMemory() = default;

    // Record a single message.
    virtual void add(const Message& m) = 0;

    // Convenience: log user + assistant exchange in one call.
    virtual void add_exchange(const std::string& user_input,
                              const std::string& ai_output);

    // Retrieve all messages in chronological order.
    virtual std::vector<Message> messages() const = 0;

    // Remove all messages.
    virtual void clear() = 0;

    // Optional session identifier for persistent memory implementations.
    // Default is a no-op for in-memory backends.
    virtual void set_session_id(const std::string& id) {}
    virtual std::string session_id() const { return {}; }
};

using MemoryPtr = std::shared_ptr<IMemory>;

// ---------------------------------------------------------------------------
// Short-term (in-memory)
// ---------------------------------------------------------------------------

// Keeps every message ever added.
class BufferMemory : public IMemory
{
public:
    void add(const Message& m) override;
    std::vector<Message> messages() const override;
    void clear() override;

private:
    mutable std::mutex mu_;
    std::vector<Message> history_;
};

// Keeps only the last `k` exchanges (k user+assistant pairs => 2k messages).
class WindowMemory : public IMemory
{
public:
    explicit WindowMemory(std::size_t k = 5);

    void add(const Message& m) override;
    std::vector<Message> messages() const override;
    void clear() override;

private:
    mutable std::mutex mu_;
    std::deque<Message> history_;
    std::size_t max_messages_;
};

// ---------------------------------------------------------------------------
// Long-term (persistent)
// ---------------------------------------------------------------------------

enum class StorageBackend
{
    JsonFile,
    Sqlite
};

// Unified persistent memory.  The actual storage backend (JSON file or SQLite)
// is selected at construction time and completely hidden behind this
// interface.  Use the static factory methods json_file() / sqlite() for
// the most common cases.
class LongTermMemory : public IMemory
{
public:
    // `path` -- file path (JSON file or SQLite DB, depending on backend).
    // `session_id` -- optional conversation thread identifier.
    explicit LongTermMemory(StorageBackend backend,
                            const std::string& path,
                            const std::string& session_id = {});
    ~LongTermMemory() override;

    // Move-only (unique_ptr member).
    LongTermMemory(LongTermMemory&&) noexcept;
    LongTermMemory& operator=(LongTermMemory&&) noexcept;

    // Non-copyable.
    LongTermMemory(const LongTermMemory&) = delete;
    LongTermMemory& operator=(const LongTermMemory&) = delete;

    // Convenience factories.
    static LongTermMemory json_file(const std::string& path,
                                    const std::string& session_id = {});

    static LongTermMemory sqlite(const std::string& db_path,
                                 const std::string& session_id = {});

    void add(const Message& m) override;
    std::vector<Message> messages() const override;
    void clear() override;

    void set_session_id(const std::string& id) override;
    std::string session_id() const override;

    // SQLite-specific: list all session IDs stored in this database.
    std::vector<std::string> list_sessions() const;

    // SQLite-specific: switch to an existing session by ID, or start a new one.
    void switch_session(const std::string& id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace memory
} // namespace langchain
