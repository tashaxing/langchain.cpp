// tests/test_gtest_utils.cpp — gtest coverage for the new util/ helpers.
#include <gtest/gtest.h>

#include "util/compress.h"
#include "util/fs.h"
#include "util/eventbus.h"
#include "util/singleton.h"
#include "util/strings.h"
#include "util/time.h"
#include "util/timer.h"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

using namespace langchain::util;

// ---------------- time ----------------

TEST(Time, MonoSecondsIsMonotonic)
{
    double a = mono_seconds();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double b = mono_seconds();
    EXPECT_GE(b, a);
}

TEST(Time, Iso8601EpochZero)
{
    EXPECT_EQ(iso8601_utc(0), "1970-01-01T00:00:00.000Z");
}

TEST(Time, Iso8601CurrentLooksRight)
{
    auto s = iso8601_utc_now();
    EXPECT_EQ(s.size(), 24u);   // YYYY-MM-DDTHH:MM:SS.mmmZ
    EXPECT_EQ(s.back(), 'Z');
}

TEST(Time, ScopedTimerReportsPositive)
{
    ScopedTimer t("self");
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    EXPECT_GT(t.elapsed_ms(), 0.0);
}

// ---------------- fs ----------------

TEST(Fs, RoundTripWriteRead)
{
    auto path = fs::join(".", "lc_fs_test_tmp.txt");
    fs::write_file(path, "hello, fs\nline2");
    EXPECT_TRUE(fs::is_file(path));
    EXPECT_EQ(fs::read_file(path), "hello, fs\nline2");
    EXPECT_EQ(fs::file_size(path), 15);
    EXPECT_TRUE(fs::remove(path));
}

TEST(Fs, MakeDirsAndPathParts)
{
    auto root = fs::join(".", "lc_fs_test_dir");
    auto nested = fs::join(root, "a/b/c");
    EXPECT_TRUE(fs::make_dirs(nested));
    EXPECT_TRUE(fs::is_dir(nested));
    EXPECT_EQ(fs::filename("/foo/bar.txt"), "bar.txt");
    EXPECT_EQ(fs::parent("/foo/bar.txt"), "/foo");
    EXPECT_EQ(fs::extension("/foo/bar.txt"), ".txt");
    EXPECT_EQ(fs::extension("/foo/no_ext"), "");
    // cleanup
    fs::remove(nested);
}

// ---------------- timer ----------------

TEST(Timer, OneShotFires)
{
    std::atomic<int> hits{0};
    Timer t;
    t.start_once(std::chrono::milliseconds(20), [&] { hits.fetch_add(1); });
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    EXPECT_EQ(hits.load(), 1);
}

TEST(Timer, IntervalFiresMultipleTimes)
{
    std::atomic<int> hits{0};
    Timer t;
    t.start_interval(std::chrono::milliseconds(20), [&] { hits.fetch_add(1); });
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    t.stop();
    EXPECT_GE(hits.load(), 3);
}

// ---------------- singleton ----------------

namespace
{
struct DemoConfig
{
    int x{42};
};
} // namespace

TEST(Singleton, ReturnsSameInstance)
{
    auto& a = Singleton<DemoConfig>::instance();
    auto& b = Singleton<DemoConfig>::instance();
    EXPECT_EQ(&a, &b);
    a.x = 7;
    EXPECT_EQ(b.x, 7);
}

// ---------------- signal ----------------

TEST(Signal, EmitInvokesAllSlots)
{
    Signal<int, const std::string&> sig;
    int sum = 0;
    std::string concat;
    auto id1 = sig.connect([&](int v, const std::string& s) { sum += v; concat += s; });
    auto id2 = sig.connect([&](int v, const std::string&) { sum += v * 10; });
    sig(2, "ab");
    EXPECT_EQ(sum, 2 + 20);
    EXPECT_EQ(concat, "ab");
    sig.disconnect(id2);
    sig(1, "x");
    EXPECT_EQ(sum, 2 + 20 + 1);
    sig.disconnect(id1);
    EXPECT_EQ(sig.slot_count(), 0u);
}

TEST(Signal, ScopedConnectionAutoDisconnects)
{
    Signal<int> sig;
    int hits = 0;
    {
        auto id = sig.connect([&](int) { ++hits; });
        ScopedConnection<Signal<int>> sc(&sig, id);
        sig(1);
        EXPECT_EQ(hits, 1);
    }
    sig(2);
    EXPECT_EQ(hits, 1); // scope ended => slot disconnected
}

TEST(EventBus, PublishesToSubscribers)
{
    auto& bus = EventBus::instance();
    bus.clear();
    int received = 0;
    auto id = bus.subscribe("metric.tick", [&](const std::any& a)
    {
        if (a.type() == typeid(int)) { received += std::any_cast<int>(a); }
    });
    bus.publish("metric.tick", std::any(5));
    bus.publish("other.topic", std::any(99));
    EXPECT_EQ(received, 5);
    EXPECT_EQ(bus.subscriber_count("metric.tick"), 1u);
    bus.unsubscribe(id);
    EXPECT_EQ(bus.subscriber_count("metric.tick"), 0u);
}

// ---------------- compress (zlib) ----------------

TEST(Compress, ZlibRoundTrip)
{
    std::string text;
    for (int i = 0; i < 100; ++i)
    {
        text += "the quick brown fox jumps over the lazy dog. ";
    }
    auto z = compress::deflate_str(text);
    EXPECT_LT(z.size(), text.size());
    EXPECT_EQ(compress::inflate_str(z), text);
}

TEST(Compress, GzipRoundTrip)
{
    std::string text = "Hello, langchain.cpp + zlib!";
    auto gz = compress::gzip(text);
    // gzip magic: 1f 8b
    ASSERT_GE(gz.size(), 2u);
    EXPECT_EQ(static_cast<unsigned char>(gz[0]), 0x1f);
    EXPECT_EQ(static_cast<unsigned char>(gz[1]), 0x8b);
    EXPECT_EQ(compress::gunzip(gz), text);
}

TEST(Compress, ZlibVersionReported)
{
    const char* v = compress::zlib_runtime_version();
    ASSERT_NE(v, nullptr);
    EXPECT_GT(std::string(v).size(), 0u);
}

// ---------------- strings (sanity) ----------------

TEST(Strings, TrimAndSplit)
{
    EXPECT_EQ(langchain::strings::trim("  hi  "), "hi");
    auto parts = langchain::strings::split("a,b,,c", ',');
    ASSERT_EQ(parts.size(), 4u);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[2], "");
    EXPECT_EQ(langchain::strings::replace_all("a-b-c", "-", "::"), "a::b::c");
}
