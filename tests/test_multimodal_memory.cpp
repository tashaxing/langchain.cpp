// tests/test_multimodal_memory.cpp — verifies that multimodal content_parts
// (text + image_base64 / image_url) round-trip through all memory backends:
//   - BufferMemory  (short-term, in-memory)
//   - WindowMemory  (short-term, sliding window)
//   - LongTermMemory SQLite (persistent)
//   - LongTermMemory JSON   (persistent)
#include <gtest/gtest.h>

#include "langchain.h"

#include <cstdio>
#include <filesystem>
#include <string>

using namespace langchain;

namespace fs = std::filesystem;

namespace
{
// Pick a unique temp path so concurrent test runs don't trample each other.
std::string make_temp_db_path(const char* tag)
{
    auto p = fs::temp_directory_path() /
             (std::string("test_mm_memory_") + tag + ".db");
    fs::remove(p);
    return p.string();
}
} // namespace

// ---------------------------------------------------------------------------
// Short-term (in-memory) — no serialization involved; Message objects are
// stored by value, so content_parts survive trivially.
// ---------------------------------------------------------------------------

TEST(MultimodalMemory, BufferMemoryRoundTrip)
{
    memory::BufferMemory mem;

    Message mm_msg = Message::user_with_image_base64(
        "Describe this image", "AQID", "image/png");
    mem.add(mm_msg);
    mem.add(Message::assistant("It is a test image."));

    auto msgs = mem.messages();
    ASSERT_EQ(msgs.size(), 2u);

    const auto& restored = msgs[0];
    EXPECT_EQ(restored.role, Role::User);
    ASSERT_EQ(restored.content_parts.size(), 2u);
    EXPECT_EQ(restored.content_parts[0].type, "text");
    EXPECT_EQ(restored.content_parts[0].text, "Describe this image");
    EXPECT_EQ(restored.content_parts[1].type, "image_base64");
    EXPECT_EQ(restored.content_parts[1].base64_data, "AQID");
    EXPECT_EQ(restored.content_parts[1].mime_type, "image/png");

    EXPECT_EQ(msgs[1].role, Role::Assistant);
}

TEST(MultimodalMemory, BufferMemoryImageUrl)
{
    memory::BufferMemory mem;

    Message mm_msg = Message::user_with_image(
        "What is this?", "https://example.com/dog.jpg");
    mem.add(mm_msg);

    auto msgs = mem.messages();
    ASSERT_EQ(msgs.size(), 1u);
    ASSERT_EQ(msgs[0].content_parts.size(), 2u);
    EXPECT_EQ(msgs[0].content_parts[1].type, "image_url");
    EXPECT_EQ(msgs[0].content_parts[1].url, "https://example.com/dog.jpg");
}

TEST(MultimodalMemory, WindowMemoryRoundTrip)
{
    memory::WindowMemory mem(3); // keep last 3 exchanges = 6 messages

    Message mm1 = Message::user_with_image_base64(
        "First image", "AA==", "image/jpeg");
    mem.add(mm1);
    mem.add(Message::assistant("Got it."));

    Message mm2 = Message::user_with_image(
        "Second image", "https://example.com/second.png");
    mem.add(mm2);
    mem.add(Message::assistant("Seen."));

    auto msgs = mem.messages();
    ASSERT_EQ(msgs.size(), 4u);

    // First user message — base64 image preserved
    ASSERT_EQ(msgs[0].content_parts.size(), 2u);
    EXPECT_EQ(msgs[0].content_parts[1].type, "image_base64");
    EXPECT_EQ(msgs[0].content_parts[1].base64_data, "AA==");
    EXPECT_EQ(msgs[0].content_parts[1].mime_type, "image/jpeg");

    // Second user message — URL image preserved
    ASSERT_EQ(msgs[2].content_parts.size(), 2u);
    EXPECT_EQ(msgs[2].content_parts[1].type, "image_url");
    EXPECT_EQ(msgs[2].content_parts[1].url, "https://example.com/second.png");
}

TEST(MultimodalMemory, WindowMemorySlidingEviction)
{
    memory::WindowMemory mem(1); // keep only 1 exchange = 2 messages

    // Add 3 exchanges; only the last should survive.
    mem.add(Message::user_with_image_base64("old", "OLD", "image/png"));
    mem.add(Message::assistant("old reply"));

    mem.add(Message::user_with_image("mid", "https://example.com/mid.png"));
    mem.add(Message::assistant("mid reply"));

    Message last_user = Message::user_with_image_base64("new", "NEW==", "image/jpeg");
    mem.add(last_user);
    mem.add(Message::assistant("new reply"));

    auto msgs = mem.messages();
    ASSERT_EQ(msgs.size(), 2u); // only last exchange kept

    // The multimodal content_parts of the surviving message must be intact.
    ASSERT_EQ(msgs[0].content_parts.size(), 2u);
    EXPECT_EQ(msgs[0].content_parts[1].type, "image_base64");
    EXPECT_EQ(msgs[0].content_parts[1].base64_data, "NEW==");
    EXPECT_EQ(msgs[0].content_parts[1].mime_type, "image/jpeg");
}

// ---------------------------------------------------------------------------
// Long-term (persistent) — exercises serialization path.
// ---------------------------------------------------------------------------

TEST(MultimodalMemory, SqliteRoundTrip)
{
    std::string db_path = make_temp_db_path("sqlite");

    Message mm_msg = Message::user_with_image_base64(
        "Describe this image",
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4//8/AwAI/AL+XJ9KgAAAAABJRU5ErkJggg==",
        "image/png");

    ASSERT_EQ(mm_msg.content_parts.size(), 2u);
    EXPECT_EQ(mm_msg.content_parts[0].type, "text");
    EXPECT_EQ(mm_msg.content_parts[1].type, "image_base64");

    // Write: multimodal user message + assistant reply.
    {
        memory::LongTermMemory mem(
            memory::LongTermMemory::sqlite(db_path, "session_a"));
        mem.add(mm_msg);
        mem.add(Message::assistant("Here is a description of the image..."));
    }

    // Read back in a fresh connection — exercises the SELECT path.
    {
        memory::LongTermMemory mem(
            memory::LongTermMemory::sqlite(db_path, "session_a"));
        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 2u);

        const auto& restored = msgs[0];
        EXPECT_EQ(restored.role, Role::User);
        ASSERT_EQ(restored.content_parts.size(), 2u);
        EXPECT_EQ(restored.content_parts[0].type, "text");
        EXPECT_EQ(restored.content_parts[0].text, "Describe this image");
        EXPECT_EQ(restored.content_parts[1].type, "image_base64");
        EXPECT_EQ(restored.content_parts[1].mime_type, "image/png");
        EXPECT_FALSE(restored.content_parts[1].base64_data.empty());

        EXPECT_EQ(msgs[1].role, Role::Assistant);
        EXPECT_EQ(msgs[1].content, "Here is a description of the image...");
    }

    fs::remove(db_path);
}

TEST(MultimodalMemory, SqliteImageUrlPart)
{
    std::string db_path = make_temp_db_path("url");

    Message mm_msg = Message::user_with_image(
        "What is in this picture?",
        "https://example.com/cat.png");

    {
        memory::LongTermMemory mem(
            memory::LongTermMemory::sqlite(db_path, "session_b"));
        mem.add(mm_msg);
    }
    {
        memory::LongTermMemory mem(
            memory::LongTermMemory::sqlite(db_path, "session_b"));
        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 1u);
        const auto& r = msgs[0];
        ASSERT_EQ(r.content_parts.size(), 2u);
        EXPECT_EQ(r.content_parts[1].type, "image_url");
        EXPECT_EQ(r.content_parts[1].url, "https://example.com/cat.png");
    }

    fs::remove(db_path);
}

TEST(MultimodalMemory, JsonFileRoundTrip)
{
    auto p = fs::temp_directory_path() / "test_mm_memory.jsonl";
    fs::remove(p);
    std::string path = p.string();

    Message mm_msg = Message::user_with_image_base64(
        "Hello", "AAAA", "image/jpeg");

    {
        memory::LongTermMemory mem(
            memory::LongTermMemory::json_file(path, "json_session"));
        mem.add(mm_msg);
    }
    {
        memory::LongTermMemory mem(
            memory::LongTermMemory::json_file(path, "json_session"));
        auto msgs = mem.messages();
        ASSERT_EQ(msgs.size(), 1u);
        const auto& r = msgs[0];
        ASSERT_EQ(r.content_parts.size(), 2u);
        EXPECT_EQ(r.content_parts[1].type, "image_base64");
        EXPECT_EQ(r.content_parts[1].base64_data, "AAAA");
        EXPECT_EQ(r.content_parts[1].mime_type, "image/jpeg");
    }

    fs::remove(p);
}
