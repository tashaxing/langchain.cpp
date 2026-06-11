// examples/23_utilities.cpp — Utility modules showcase.
//
// Demonstrates:
//   1. Logging: init, set_level, macro usage.
//   2. Timer: one-shot and interval modes.
//   3. EventBus: process-wide pub/sub with std::any.
//   4. Compress: gzip / gunzip round-trip.
//   5. String helpers: trim, split, replace_all.
//   6. Filesystem helpers: read_file, write_file, list_dir.
//
// No network required.
#include "langchain.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

void demo_logging()
{
    std::cout << "=== Logging ===\n";
    std::filesystem::create_directories("build/logs");

    langchain::log::init("build/logs", "demo.log",
                         spdlog::level::debug, 1024 * 1024, 3, true);

    LOG_DEBUG("This is a debug message: {}", 42);
    LOG_INFO("This is an info message");
    LOG_WARN("This is a warning");
    LOG_ERROR("This is an error: {}", "something went wrong");

    // Change level at runtime
    langchain::log::set_level(spdlog::level::warn);
    LOG_DEBUG("This debug message should NOT appear");
    LOG_WARN("This warning SHOULD appear");

    std::cout << "  Log files written to build/logs/demo.log\n";
}

void demo_timer()
{
    std::cout << "\n=== Timer ===\n";
    langchain::util::Timer timer;

    // One-shot
    std::atomic<bool> fired{false};
    timer.start_once(std::chrono::milliseconds(100),
        [&fired]()
    {
        fired.store(true);
        std::cout << "  One-shot timer fired!\n";
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "  One-shot fired? " << (fired.load() ? "yes" : "no") << "\n";

    // Interval
    std::atomic<int> count{0};
    timer.start_interval(std::chrono::milliseconds(50),
        [&count]()
    {
        ++count;
        std::cout << "  Interval tick " << count.load() << "\n";
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(180));
    timer.stop();
    std::cout << "  Total ticks: " << count.load() << "\n";
}

void demo_eventbus()
{
    std::cout << "\n=== EventBus ===\n";
    auto& bus = langchain::util::EventBus::instance();
    bus.clear();

    langchain::util::SlotId id = bus.subscribe("greeting",
        [](const std::any& payload)
    {
        try
        {
            auto msg = std::any_cast<std::string>(payload);
            std::cout << "  Received greeting: " << msg << "\n";
        }
        catch (const std::bad_any_cast&)
        {
            std::cout << "  Received unknown payload type\n";
        }
    });

    std::cout << "  Subscribers to 'greeting': "
              << bus.subscriber_count("greeting") << "\n";

    bus.publish("greeting", std::string("Hello from EventBus!"));
    bus.publish("other_topic", std::string("This won't be received"));

    bus.unsubscribe(id);
    std::cout << "  After unsubscribe: "
              << bus.subscriber_count("greeting") << "\n";
}

void demo_compress()
{
    std::cout << "\n=== Compress ===\n";
    std::string original =
        "C++20 introduces modules, coroutines, concepts, and ranges. "
        "These features fundamentally modernize how we write C++. "
        "Modules replace header files and improve compilation times. "
        "Coroutines enable efficient asynchronous programming.";

    std::cout << "  Original size: " << original.size() << " bytes\n";

    std::string compressed = langchain::util::compress::gzip(original);
    std::cout << "  Gzip size:     " << compressed.size() << " bytes\n";

    std::string decompressed = langchain::util::compress::gunzip(compressed);
    std::cout << "  Decompressed matches: "
              << (decompressed == original ? "yes" : "no") << "\n";

    std::cout << "  zlib version: " << langchain::util::compress::zlib_runtime_version() << "\n";
}

void demo_strings()
{
    std::cout << "\n=== String helpers ===\n";
    std::string s = "  hello world  ";
    std::cout << "  trim('" << s << "') = '" << langchain::strings::trim(s) << "'\n";

    std::string csv = "a,b,c,d";
    auto parts = langchain::strings::split(csv, ',');
    std::cout << "  split('" << csv << "', ',') = ";
    for (const auto& p : parts)
    {
        std::cout << "'" << p << "' ";
    }
    std::cout << "\n";

    std::string text = "foo bar foo baz foo";
    std::cout << "  replace_all('" << text << "', 'foo', 'X') = '"
              << langchain::strings::replace_all(text, "foo", "X") << "'\n";

    std::cout << "  starts_with('hello', 'he') = "
              << (langchain::strings::starts_with("hello", "he") ? "true" : "false") << "\n";
    std::cout << "  contains('hello', 'll') = "
              << (langchain::strings::contains("hello", "ll") ? "true" : "false") << "\n";
}

void demo_fs()
{
    std::cout << "\n=== Filesystem helpers ===\n";
    std::filesystem::create_directories("build/fs_demo");

    std::string path = "build/fs_demo/test.txt";
    langchain::util::fs::write_file(path, "Hello from fs helper!");
    std::cout << "  Wrote: " << path << "\n";

    std::string content = langchain::util::fs::read_file(path);
    std::cout << "  Read back: '" << content << "'\n";

    std::cout << "  exists: "
              << (langchain::util::fs::exists(path) ? "yes" : "no") << "\n";
    std::cout << "  is_file: "
              << (langchain::util::fs::is_file(path) ? "yes" : "no") << "\n";
    std::cout << "  file_size: "
              << langchain::util::fs::file_size(path) << " bytes\n";
    std::cout << "  extension: "
              << langchain::util::fs::extension(path) << "\n";
    std::cout << "  filename: "
              << langchain::util::fs::filename(path) << "\n";
    std::cout << "  parent: "
              << langchain::util::fs::parent(path) << "\n";

    auto entries = langchain::util::fs::list_dir("build/fs_demo");
    std::cout << "  list_dir: ";
    for (const auto& e : entries)
    {
        std::cout << e << " ";
    }
    std::cout << "\n";

    langchain::util::fs::remove(path);
    std::cout << "  After remove, exists: "
              << (langchain::util::fs::exists(path) ? "yes" : "no") << "\n";

    std::filesystem::remove_all("build/fs_demo");
}

int main()
{
    try
    {
        demo_logging();
        demo_timer();
        demo_eventbus();
        demo_compress();
        demo_strings();
        demo_fs();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nDone.\n";
    return 0;
}
