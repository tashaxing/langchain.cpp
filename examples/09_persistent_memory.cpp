// examples/09_persistent_memory.cpp -- demonstrate short-term and long-term
// memory: BufferMemory, WindowMemory, and LongTermMemory with JSON/SQLite
// backends.
//
// Run:
//   ./build/09_persistent_memory              # Linux/macOS
//   ./build_x64/Debug/09_persistent_memory    # Windows
#include "langchain.h"

#include <filesystem>
#include <iostream>
#include <thread>

void demo_short_term()
{
    std::cout << "=== Short-term memory ===\n";

    // BufferMemory: keeps everything
    langchain::memory::BufferMemory buf;
    buf.add(langchain::Message::user("Hello"));
    buf.add(langchain::Message::assistant("Hi there!"));
    buf.add(langchain::Message::user("How are you?"));
    std::cout << "BufferMemory has " << buf.messages().size() << " messages\n";

    // WindowMemory: keeps only last k exchanges
    langchain::memory::WindowMemory win(1);  // 1 exchange = 2 messages
    win.add(langchain::Message::user("A"));
    win.add(langchain::Message::assistant("Reply to A"));
    win.add(langchain::Message::user("B"));
    win.add(langchain::Message::assistant("Reply to B"));
    win.add(langchain::Message::user("C"));
    std::cout << "WindowMemory(1) has " << win.messages().size()
              << " messages (should be 2, last exchange)\n";
}

void demo_json_memory()
{
    std::cout << "\n=== LongTermMemory (JSON file) ===\n";

    std::filesystem::create_directories("build");
    std::string path = "build/memory_test.json";

    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::json_file(path, "session-abc"));
        mem.add(langchain::Message::user("What is C++?"));
        mem.add(langchain::Message::assistant(
            "C++ is a general-purpose programming language."));
        std::cout << "Wrote 2 messages to " << path << "\n";
    }

    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::json_file(path, "session-abc"));
        auto msgs = mem.messages();
        std::cout << "Read back " << msgs.size() << " messages:\n";
        for (const auto& m : msgs)
        {
            std::cout << "  [" << langchain::to_string(m.role) << "] "
                      << m.content << "\n";
        }
    }
}

void demo_sqlite_memory()
{
    std::cout << "\n=== LongTermMemory (SQLite) ===\n";

    std::filesystem::create_directories("build");
    std::string db_path = "build/memory_test.db";

    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::sqlite(db_path, "chat-session-1"));
        mem.add(langchain::Message::user("Tell me a joke"));
        mem.add(langchain::Message::assistant(
            "Why did the chicken cross the road?"));
        mem.add(langchain::Message::user("I don't know"));
        mem.add(langchain::Message::assistant("To get to the other side!"));
        std::cout << "Wrote 4 messages to " << db_path << "\n";
    }

    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::sqlite(db_path, "chat-session-1"));
        auto msgs = mem.messages();
        std::cout << "Read back " << msgs.size() << " messages:\n";
        for (const auto& m : msgs)
        {
            std::cout << "  [" << langchain::to_string(m.role) << "] "
                      << m.content << "\n";
        }
    }

    // Switch to a different session
    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::sqlite(db_path));
        auto sessions = mem.list_sessions();
        std::cout << "Sessions in DB: " << sessions.size() << "\n";
        for (const auto& s : sessions)
        {
            std::cout << "  - " << s << "\n";
        }
    }
}

int main()
{
    using namespace langchain;

    try
    {
        demo_short_term();
        demo_json_memory();
        demo_sqlite_memory();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nDone.\n";
    return 0;
}
