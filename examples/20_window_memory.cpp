// examples/20_window_memory.cpp — WindowMemory and session management.
//
// Demonstrates:
//   1. BufferMemory vs WindowMemory: eviction behavior.
//   2. WindowMemory with different k values.
//   3. LongTermMemory session switching and listing.
//
// No network required.
#include "langchain.h"

#include <filesystem>
#include <iostream>

void demo_buffer_vs_window()
{
    std::cout << "=== BufferMemory (keeps everything) ===\n";
    langchain::memory::BufferMemory buf;
    buf.add(langchain::Message::user("Q1"));
    buf.add(langchain::Message::assistant("A1"));
    buf.add(langchain::Message::user("Q2"));
    buf.add(langchain::Message::assistant("A2"));
    buf.add(langchain::Message::user("Q3"));
    std::cout << "Messages: " << buf.messages().size() << " (expected 5)\n";

    std::cout << "\n=== WindowMemory(k=1, keeps last 1 exchange = 2 messages) ===\n";
    langchain::memory::WindowMemory win(1);
    win.add(langchain::Message::user("Q1"));
    win.add(langchain::Message::assistant("A1"));
    win.add(langchain::Message::user("Q2"));
    win.add(langchain::Message::assistant("A2"));
    win.add(langchain::Message::user("Q3"));
    auto msgs = win.messages();
    std::cout << "Messages: " << msgs.size() << " (expected 2)\n";
    for (const auto& m : msgs)
    {
        std::cout << "  [" << langchain::to_string(m.role) << "] " << m.content << "\n";
    }

    std::cout << "\n=== WindowMemory(k=2, keeps last 2 exchanges = 4 messages) ===\n";
    langchain::memory::WindowMemory win2(2);
    win2.add(langchain::Message::user("Q1"));
    win2.add(langchain::Message::assistant("A1"));
    win2.add(langchain::Message::user("Q2"));
    win2.add(langchain::Message::assistant("A2"));
    win2.add(langchain::Message::user("Q3"));
    win2.add(langchain::Message::assistant("A3"));
    auto msgs2 = win2.messages();
    std::cout << "Messages: " << msgs2.size() << " (expected 4)\n";
    for (const auto& m : msgs2)
    {
        std::cout << "  [" << langchain::to_string(m.role) << "] " << m.content << "\n";
    }
}

void demo_session_management()
{
    std::cout << "\n=== LongTermMemory session management ===\n";
    std::filesystem::create_directories("build");
    std::string db = "build/window_memory_demo.db";

    // Create two sessions
    {
        langchain::memory::LongTermMemory m1(
            langchain::memory::LongTermMemory::sqlite(db, "user-alice"));
        m1.add(langchain::Message::user("Alice's question"));
        m1.add(langchain::Message::assistant("Answer to Alice"));

        langchain::memory::LongTermMemory m2(
            langchain::memory::LongTermMemory::sqlite(db, "user-bob"));
        m2.add(langchain::Message::user("Bob's question"));
        m2.add(langchain::Message::assistant("Answer to Bob"));
    }

    // List sessions
    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::sqlite(db, "user-alice"));
        auto sessions = mem.list_sessions();
        std::cout << "Sessions: " << sessions.size() << "\n";
        for (const auto& s : sessions)
        {
            std::cout << "  - " << s << "\n";
        }
    }

    // Switch session
    {
        langchain::memory::LongTermMemory mem(
            langchain::memory::LongTermMemory::sqlite(db, "user-alice"));
        std::cout << "\nBefore switch: " << mem.messages().size()
                  << " messages (Alice)\n";
        mem.switch_session("user-bob");
        std::cout << "After switch:  " << mem.messages().size()
                  << " messages (Bob)\n";
    }

    std::filesystem::remove(db);
}

int main()
{
    try
    {
        demo_buffer_vs_window();
        demo_session_management();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nDone.\n";
    return 0;
}
