// tests/test_memory.cpp — Memory unit tests.
//
// Covers BufferMemory, WindowMemory, LongTermMemory (JSON + SQLite backends),
// session switching, and multimodal round-trip.
#include <gtest/gtest.h>

#include "memory/memory.h"
#include "util/fs.h"

#include <chrono>
#include <cstdio>
#include <filesystem>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string make_temp_path(const std::string& suffix)
{
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = std::filesystem::temp_directory_path() /
                ("lc_test_" + std::to_string(now) + suffix);
    return path.string();
}

void remove_file(const std::string& path)
{
    std::remove(path.c_str());
}

} // namespace

// ---------------------------------------------------------------------------
// BufferMemory
// ---------------------------------------------------------------------------
TEST(BufferMemory, KeepsAllMessages)
{
    memory::BufferMemory mem;
    mem.add(Message::user("u1"));
    mem.add(Message::assistant("a1"));
    mem.add(Message::user("u2"));

    auto msgs = mem.messages();
    ASSERT_EQ(msgs.size(), 3u);
    EXPECT_EQ(msgs[0].content, "u1");
    EXPECT_EQ(msgs[1].content, "a1");
    EXPECT_EQ(msgs[2].content, "u2");
}

TEST(BufferMemory, ClearRemovesAll)
{
    memory::BufferMemory mem;
    mem.add(Message::user("x"));
    mem.clear();
    EXPECT_TRUE(mem.messages().empty());
}

TEST(BufferMemory, AddExchange)
{
    memory::BufferMemory mem;
    mem.add_exchange("hello", "hi there");

    auto msgs = mem.messages();
    ASSERT_EQ(msgs.size(), 2u);
    EXPECT_EQ(msgs[0].role, Role::User);
    EXPECT_EQ(msgs[1].role, Role::Assistant);
}

// ---------------------------------------------------------------------------
// WindowMemory
// ---------------------------------------------------------------------------
TEST(WindowMemory, EvictsOldExchanges)
{
    memory::WindowMemory w(/*k=*/2);
    w.add_exchange("u1", "a1");
    w.add_exchange("u2", "a2");
    w.add_exchange("u3", "a3"); // should evict u1/a1

    auto msgs = w.messages();
    EXPECT_EQ(msgs.size(), 4u);
    EXPECT_EQ(msgs.front().content, "u2");
}

TEST(WindowMemory, Clear)
{
    memory::WindowMemory w(5);
    w.add(Message::user("x"));
    w.clear();
    EXPECT_TRUE(w.messages().empty());
}

// ---------------------------------------------------------------------------
// LongTermMemory — JSON backend
// ---------------------------------------------------------------------------
TEST(LongTermMemory, JsonRoundTrip)
{
    std::string path = make_temp_path(".jsonl");
    {
        memory::LongTermMemory mem = memory::LongTermMemory::json_file(path, "sess1");
        mem.add(Message::user("hello"));
        mem.add(Message::assistant("world"));

        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 2u);
        EXPECT_EQ(msgs[0].content, "hello");
    }
    // Re-open and verify persistence.
    {
        memory::LongTermMemory mem = memory::LongTermMemory::json_file(path, "sess1");
        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 2u);
        EXPECT_EQ(msgs[1].content, "world");
    }
    remove_file(path);
}

TEST(LongTermMemory, JsonSessionIsolation)
{
    std::string path = make_temp_path(".jsonl");
    {
        memory::LongTermMemory m1 = memory::LongTermMemory::json_file(path, "A");
        m1.add(Message::user("from A"));

        memory::LongTermMemory m2 = memory::LongTermMemory::json_file(path, "B");
        m2.add(Message::user("from B"));
    }
    {
        memory::LongTermMemory ma = memory::LongTermMemory::json_file(path, "A");
        EXPECT_EQ(ma.messages().size(), 1u);
        EXPECT_EQ(ma.messages()[0].content, "from A");

        memory::LongTermMemory mb = memory::LongTermMemory::json_file(path, "B");
        EXPECT_EQ(mb.messages().size(), 1u);
        EXPECT_EQ(mb.messages()[0].content, "from B");
    }
    remove_file(path);
}

TEST(LongTermMemory, JsonClear)
{
    std::string path = make_temp_path(".jsonl");
    memory::LongTermMemory mem = memory::LongTermMemory::json_file(path, "s");
    mem.add(Message::user("x"));
    mem.clear();
    EXPECT_TRUE(mem.messages().empty());
    remove_file(path);
}

// ---------------------------------------------------------------------------
// LongTermMemory — SQLite backend
// ---------------------------------------------------------------------------
TEST(LongTermMemory, SqliteRoundTrip)
{
    std::string path = make_temp_path(".db");
    {
        memory::LongTermMemory mem = memory::LongTermMemory::sqlite(path, "sess1");
        mem.add(Message::user("sqlite hello"));
        mem.add(Message::assistant("sqlite world"));

        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 2u);
    }
    {
        memory::LongTermMemory mem = memory::LongTermMemory::sqlite(path, "sess1");
        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 2u);
        EXPECT_EQ(msgs[0].content, "sqlite hello");
    }
    remove_file(path);
}

TEST(LongTermMemory, SqliteListSessions)
{
    std::string path = make_temp_path(".db");
    {
        memory::LongTermMemory m1 = memory::LongTermMemory::sqlite(path, "alpha");
        m1.add(Message::user("a"));

        memory::LongTermMemory m2 = memory::LongTermMemory::sqlite(path, "beta");
        m2.add(Message::user("b"));
    }
    {
        memory::LongTermMemory mem = memory::LongTermMemory::sqlite(path, "alpha");
        auto sessions = mem.list_sessions();
        EXPECT_EQ(sessions.size(), 2u);
    }
    remove_file(path);
}

TEST(LongTermMemory, SqliteSwitchSession)
{
    std::string path = make_temp_path(".db");
    {
        memory::LongTermMemory mem = memory::LongTermMemory::sqlite(path, "orig");
        mem.add(Message::user("orig-msg"));
        mem.switch_session("new");
        mem.add(Message::user("new-msg"));

        EXPECT_EQ(mem.messages().size(), 1u);
        EXPECT_EQ(mem.messages()[0].content, "new-msg");
    }
    remove_file(path);
}

TEST(LongTermMemory, SqliteClear)
{
    std::string path = make_temp_path(".db");
    memory::LongTermMemory mem = memory::LongTermMemory::sqlite(path, "s");
    mem.add(Message::user("x"));
    mem.clear();
    EXPECT_TRUE(mem.messages().empty());
    remove_file(path);
}

// ---------------------------------------------------------------------------
// LongTermMemory — multimodal round-trip
// ---------------------------------------------------------------------------
TEST(LongTermMemory, MultimodalRoundTrip)
{
    std::string path = make_temp_path(".db");
    {
        memory::LongTermMemory mem = memory::LongTermMemory::sqlite(path, "mm");
        Message m = Message::user_with_image("describe this", "http://example.com/img.png");
        mem.add(m);

        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 1u);
        // content is empty when content_parts is used; content_parts carries the text.
        ASSERT_EQ(msgs[0].content_parts.size(), 2u);
        EXPECT_EQ(msgs[0].content_parts[0].type, "text");
        EXPECT_EQ(msgs[0].content_parts[0].text, "describe this");
        EXPECT_EQ(msgs[0].content_parts[1].type, "image_url");
        EXPECT_EQ(msgs[0].content_parts[1].url, "http://example.com/img.png");
    }
    remove_file(path);
}
