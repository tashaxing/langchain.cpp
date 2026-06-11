// src/memory/memory.cpp -- implementations for conversation memory classes.
#include "memory/memory.h"

#include "util/fs.h"

#include <sqlite3.h>

#include <fstream>
#include <iomanip>
#include <sstream>

namespace langchain
{
namespace memory
{

// ---------------------------------------------------------------------------
// IMemory
// ---------------------------------------------------------------------------

void IMemory::add_exchange(const std::string& user_input,
                           const std::string& ai_output)
{
    add(Message::user(user_input));
    add(Message::assistant(ai_output));
}

// ---------------------------------------------------------------------------
// BufferMemory
// ---------------------------------------------------------------------------

void BufferMemory::add(const Message& m)
{
    std::lock_guard<std::mutex> lk(mu_);
    history_.push_back(m);
}

std::vector<Message> BufferMemory::messages() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return history_;
}

void BufferMemory::clear()
{
    std::lock_guard<std::mutex> lk(mu_);
    history_.clear();
}

// ---------------------------------------------------------------------------
// WindowMemory
// ---------------------------------------------------------------------------

WindowMemory::WindowMemory(std::size_t k)
    : max_messages_(k * 2)
{
}

void WindowMemory::add(const Message& m)
{
    std::lock_guard<std::mutex> lk(mu_);
    history_.push_back(m);
    while (history_.size() > max_messages_)
    {
        history_.pop_front();
    }
}

std::vector<Message> WindowMemory::messages() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return std::vector<Message>(history_.begin(), history_.end());
}

void WindowMemory::clear()
{
    std::lock_guard<std::mutex> lk(mu_);
    history_.clear();
}

// ---------------------------------------------------------------------------
// LongTermMemory -- internal implementations
// ---------------------------------------------------------------------------

namespace
{

// Encode a single Message as a JSON object. Includes multimodal `content_parts`
// when present, so persistent memory can round-trip image data (base64) along
// with text.
json encode_message(const Message& m)
{
    json j;
    j["role"]    = to_string(m.role);
    j["content"] = m.content;
    if (!m.content_parts.empty())
    {
        json arr = json::array();
        for (const auto& p : m.content_parts)
        {
            json jp;
            jp["type"] = p.type;
            if (!p.text.empty())        jp["text"]        = p.text;
            if (!p.url.empty())         jp["url"]         = p.url;
            if (!p.base64_data.empty()) jp["base64_data"] = p.base64_data;
            if (!p.mime_type.empty())   jp["mime_type"]   = p.mime_type;
            arr.push_back(std::move(jp));
        }
        j["content_parts"] = std::move(arr);
    }
    if (!m.name.empty())
    {
        j["name"] = m.name;
    }
    if (!m.tool_calls.empty())
    {
        json arr = json::array();
        for (const auto& tc : m.tool_calls)
        {
            arr.push_back({
                {"id", tc.id},
                {"name", tc.name},
                {"arguments", tc.arguments}
            });
        }
        j["tool_calls"] = std::move(arr);
    }
    if (!m.tool_call_id.empty())
    {
        j["tool_call_id"] = m.tool_call_id;
    }
    return j;
}

// Decode a JSON object into a Message. Mirrors encode_message() and recovers
// multimodal `content_parts` if present.
Message decode_message(const json& j)
{
    Message m;
    if (j.contains("role"))
    {
        m.role = role_from_string(j["role"].get<std::string>());
    }
    if (j.contains("content"))
    {
        m.content = j["content"].get<std::string>();
    }
    if (j.contains("content_parts") && j["content_parts"].is_array())
    {
        for (const auto& jp : j["content_parts"])
        {
            ContentPart p;
            p.type        = jp.value("type",        std::string());
            p.text        = jp.value("text",        std::string());
            p.url         = jp.value("url",         std::string());
            p.base64_data = jp.value("base64_data", std::string());
            p.mime_type   = jp.value("mime_type",   std::string());
            m.content_parts.push_back(std::move(p));
        }
    }
    if (j.contains("name"))
    {
        m.name = j["name"].get<std::string>();
    }
    if (j.contains("tool_calls") && j["tool_calls"].is_array())
    {
        for (const auto& tc : j["tool_calls"])
        {
            ToolCall t;
            t.id        = tc.value("id", std::string());
            t.name      = tc.value("name", std::string());
            t.arguments = tc.value("arguments", std::string());
            m.tool_calls.push_back(std::move(t));
        }
    }
    if (j.contains("tool_call_id"))
    {
        m.tool_call_id = j["tool_call_id"].get<std::string>();
    }
    return m;
}

// RAII wrapper for sqlite3_stmt to avoid manual finalize in every path.
struct SqliteStmt
{
    sqlite3_stmt* stmt = nullptr;
    explicit SqliteStmt(sqlite3_stmt* s) : stmt(s) {}
    ~SqliteStmt() { if (stmt) sqlite3_finalize(stmt); }
    sqlite3_stmt* operator*() const { return stmt; }
};

// Execute a simple SQL statement (no results). Returns true on success.
bool sqlite_exec(sqlite3* db, const std::string& sql)
{
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
    return rc == SQLITE_OK;
}

} // namespace

// ---------------------------------------------------------------------------
// LongTermMemory::Impl -- holds either JsonFile or SQLite backend.
// ---------------------------------------------------------------------------

struct LongTermMemory::Impl
{
    StorageBackend backend;

    // JsonFile state
    mutable std::mutex json_mu;
    std::string json_path;
    std::string json_session_id;
    mutable std::vector<Message> json_cache;
    mutable bool json_dirty = false;

    // SQLite state
    sqlite3* db = nullptr;
    mutable std::mutex sqlite_mu;
    std::string sqlite_db_path;
    std::string sqlite_session_id;

    Impl(StorageBackend b,
         const std::string& path,
         const std::string& session_id)
        : backend(b)
    {
        if (backend == StorageBackend::JsonFile)
        {
            json_path = path;
            json_session_id = session_id;
            load_json_();
        }
        else
        {
            sqlite_db_path = path;
            sqlite_session_id = session_id;
            open_sqlite_();
        }
    }

    ~Impl()
    {
        if (backend == StorageBackend::Sqlite && db)
        {
            sqlite3_close(db);
        }
    }

    // ---- JsonFile backend ----

    void load_json_() const
    {
        json_cache.clear();
        if (!util::fs::is_file(json_path))
        {
            return;
        }

        std::ifstream ifs(json_path);
        if (!ifs.is_open())
        {
            return;
        }

        std::string line;
        while (std::getline(ifs, line))
        {
            if (line.empty())
            {
                continue;
            }
            json j = json::parse(line, nullptr, false);
            if (j.is_discarded())
            {
                continue;
            }
            if (j.contains("session_id") &&
                j["session_id"].get<std::string>() != json_session_id)
            {
                continue;
            }
            json_cache.push_back(decode_message(j));
        }
        json_dirty = false;
    }

    void append_json_(const Message& m) const
    {
        json j = encode_message(m);
        j["timestamp"]  = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        j["session_id"] = json_session_id;

        std::ofstream ofs(json_path, std::ios::app);
        if (ofs.is_open())
        {
            ofs << j.dump() << "\n";
        }
    }

    void rewrite_json_() const
    {
        std::ofstream ofs(json_path);
        if (!ofs.is_open())
        {
            return;
        }
        for (const auto& m : json_cache)
        {
            json j = encode_message(m);
            j["timestamp"]  = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            j["session_id"] = json_session_id;
            ofs << j.dump() << "\n";
        }
    }

    // ---- SQLite backend ----

    void open_sqlite_()
    {
        int rc = sqlite3_open(sqlite_db_path.c_str(), &db);
        if (rc != SQLITE_OK || !db)
        {
            throw LCError("LongTermMemory: cannot open SQLite database " + sqlite_db_path);
        }

        sqlite_exec(db, R"(
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at INTEGER DEFAULT (strftime('%s','now'))
            );
        )");

        sqlite_exec(db, R"(
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                name TEXT,
                tool_call_id TEXT,
                created_at INTEGER DEFAULT (strftime('%s','now'))
            );
        )");

        // Backward-compat migration: add `payload` column on databases created
        // before multimodal support landed. ALTER TABLE fails harmlessly when
        // the column already exists; we ignore the error code.
        sqlite_exec(db, "ALTER TABLE messages ADD COLUMN payload TEXT;");

        if (!sqlite_session_id.empty())
        {
            std::string sql = "INSERT OR IGNORE INTO sessions (id) VALUES ('" +
                              sqlite_session_id + "');";
            sqlite_exec(db, sql);
        }
    }

    // ---- Public interface helpers ----

    void add(const Message& m)
    {
        if (backend == StorageBackend::JsonFile)
        {
            std::lock_guard<std::mutex> lk(json_mu);
            json_cache.push_back(m);
            json_dirty = true;
            append_json_(m);
        }
        else
        {
            std::lock_guard<std::mutex> lk(sqlite_mu);

            const char* sql =
                "INSERT INTO messages (session_id, role, content, name, tool_call_id, payload) "
                "VALUES (?, ?, ?, ?, ?, ?);";

            sqlite3_stmt* stmt = nullptr;
            int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
            if (rc != SQLITE_OK || !stmt)
            {
                throw LCError("LongTermMemory: SQLite prepare failed");
            }
            SqliteStmt guard(stmt);

            // Full Message JSON for payload column — enables round-tripping
            // multimodal content_parts and tool_calls without schema churn.
            std::string role_str   = to_string(m.role);
            std::string payload_js = encode_message(m).dump();

            sqlite3_bind_text(stmt, 1, sqlite_session_id.c_str(),
                              static_cast<int>(sqlite_session_id.size()), SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, role_str.c_str(),
                              static_cast<int>(role_str.size()), SQLITE_STATIC);
            sqlite3_bind_text(stmt, 3, m.content.c_str(),
                              static_cast<int>(m.content.size()), SQLITE_STATIC);
            sqlite3_bind_text(stmt, 4, m.name.c_str(),
                              static_cast<int>(m.name.size()), SQLITE_STATIC);
            sqlite3_bind_text(stmt, 5, m.tool_call_id.c_str(),
                              static_cast<int>(m.tool_call_id.size()), SQLITE_STATIC);
            sqlite3_bind_text(stmt, 6, payload_js.c_str(),
                              static_cast<int>(payload_js.size()), SQLITE_TRANSIENT);

            sqlite3_step(stmt);
        }
    }

    std::vector<Message> messages() const
    {
        if (backend == StorageBackend::JsonFile)
        {
            std::lock_guard<std::mutex> lk(json_mu);
            if (json_dirty)
            {
                load_json_();
            }
            return json_cache;
        }
        else
        {
            std::lock_guard<std::mutex> lk(sqlite_mu);

            std::vector<Message> out;
            // Prefer `payload` (full JSON) when available; fall back to the
            // legacy columns for rows written by older versions.
            const char* sql =
                "SELECT role, content, name, tool_call_id, payload FROM messages "
                "WHERE session_id = ? ORDER BY id ASC;";

            sqlite3_stmt* stmt = nullptr;
            int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
            if (rc != SQLITE_OK || !stmt)
            {
                return out;
            }
            SqliteStmt guard(stmt);

            sqlite3_bind_text(stmt, 1, sqlite_session_id.c_str(),
                              static_cast<int>(sqlite_session_id.size()), SQLITE_STATIC);

            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                // Try payload first (contains full Message JSON).
                if (sqlite3_column_type(stmt, 4) == SQLITE_TEXT)
                {
                    const char* raw = reinterpret_cast<const char*>(
                        sqlite3_column_text(stmt, 4));
                    if (raw && *raw)
                    {
                        json j = json::parse(raw, nullptr, false);
                        if (!j.is_discarded())
                        {
                            out.push_back(decode_message(j));
                            continue;
                        }
                    }
                }

                // Fallback: legacy columns (pre-migration rows).
                Message m;
                m.role    = role_from_string(
                    reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
                m.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                const char* tcid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
                if (name) m.name = name;
                if (tcid) m.tool_call_id = tcid;
                out.push_back(std::move(m));
            }
            return out;
        }
    }

    void clear()
    {
        if (backend == StorageBackend::JsonFile)
        {
            std::lock_guard<std::mutex> lk(json_mu);
            json_cache.clear();
            json_dirty = false;
            rewrite_json_();
        }
        else
        {
            std::lock_guard<std::mutex> lk(sqlite_mu);
            std::string sql =
                "DELETE FROM messages WHERE session_id = '" + sqlite_session_id + "';";
            sqlite_exec(db, sql);
        }
    }

    void set_session_id(const std::string& id)
    {
        if (backend == StorageBackend::JsonFile)
        {
            std::lock_guard<std::mutex> lk(json_mu);
            json_session_id = id;
            json_dirty = true;
        }
        else
        {
            std::lock_guard<std::mutex> lk(sqlite_mu);
            sqlite_session_id = id;
        }
    }

    std::string session_id() const
    {
        if (backend == StorageBackend::JsonFile)
        {
            std::lock_guard<std::mutex> lk(json_mu);
            return json_session_id;
        }
        else
        {
            std::lock_guard<std::mutex> lk(sqlite_mu);
            return sqlite_session_id;
        }
    }

    std::vector<std::string> list_sessions() const
    {
        if (backend != StorageBackend::Sqlite)
        {
            return {};
        }

        std::lock_guard<std::mutex> lk(sqlite_mu);

        std::vector<std::string> out;
        const char* sql = "SELECT id FROM sessions ORDER BY created_at DESC;";

        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK || !stmt)
        {
            return out;
        }
        SqliteStmt guard(stmt);

        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* s = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (s) out.emplace_back(s);
        }
        return out;
    }

    void switch_session(const std::string& id)
    {
        if (backend != StorageBackend::Sqlite)
        {
            return;
        }

        std::lock_guard<std::mutex> lk(sqlite_mu);
        sqlite_session_id = id;

        std::string sql = "INSERT OR IGNORE INTO sessions (id) VALUES ('" + id + "');";
        sqlite_exec(db, sql);
    }
};

// ---------------------------------------------------------------------------
// LongTermMemory
// ---------------------------------------------------------------------------

LongTermMemory::LongTermMemory(StorageBackend backend,
                               const std::string& path,
                               const std::string& session_id)
    : impl_(std::make_unique<Impl>(backend, path, session_id))
{
}

LongTermMemory::~LongTermMemory() = default;

LongTermMemory::LongTermMemory(LongTermMemory&&) noexcept = default;
LongTermMemory& LongTermMemory::operator=(LongTermMemory&&) noexcept = default;

LongTermMemory LongTermMemory::json_file(const std::string& path,
                                         const std::string& session_id)
{
    return LongTermMemory(StorageBackend::JsonFile, path, session_id);
}

LongTermMemory LongTermMemory::sqlite(const std::string& db_path,
                                      const std::string& session_id)
{
    return LongTermMemory(StorageBackend::Sqlite, db_path, session_id);
}

void LongTermMemory::add(const Message& m)
{
    impl_->add(m);
}

std::vector<Message> LongTermMemory::messages() const
{
    return impl_->messages();
}

void LongTermMemory::clear()
{
    impl_->clear();
}

void LongTermMemory::set_session_id(const std::string& id)
{
    impl_->set_session_id(id);
}

std::string LongTermMemory::session_id() const
{
    return impl_->session_id();
}

std::vector<std::string> LongTermMemory::list_sessions() const
{
    return impl_->list_sessions();
}

void LongTermMemory::switch_session(const std::string& id)
{
    impl_->switch_session(id);
}

} // namespace memory
} // namespace langchain
