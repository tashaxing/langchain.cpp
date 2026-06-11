// src/util/time.cpp
#include "util/time.h"
#include "util/logging.h"

#include <cstdio>
#include <ctime>

namespace langchain
{
namespace util
{

namespace
{

// Process-start anchor for mono_seconds().
const auto kMonoStart = std::chrono::steady_clock::now();

} // namespace

std::int64_t now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

double mono_seconds()
{
    using namespace std::chrono;
    auto now = steady_clock::now();
    return duration_cast<duration<double>>(now - kMonoStart).count();
}

std::string iso8601_utc(std::int64_t epoch_ms)
{
    std::time_t sec = static_cast<std::time_t>(epoch_ms / 1000);
    int ms          = static_cast<int>(epoch_ms % 1000);
    if (ms < 0)
    {
        ms += 1000;
        --sec;
    }

    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &sec);
#else
    gmtime_r(&sec, &tm);
#endif

    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
    return std::string(buf);
}

std::string iso8601_utc_now()
{
    return iso8601_utc(now_ms());
}

// ---------------- ScopedTimer ----------------

ScopedTimer::ScopedTimer(std::string label)
    : label_(std::move(label)),
      start_(std::chrono::steady_clock::now())
{
}

ScopedTimer::~ScopedTimer()
{
    if (consumed_)
    {
        return;
    }
    auto end = std::chrono::steady_clock::now();
    auto ms  = std::chrono::duration<double, std::milli>(end - start_).count();
    LOG_DEBUG("[timer] {} : {} ms", label_.empty() ? "(unnamed)" : label_, ms);
}

double ScopedTimer::elapsed_ms()
{
    consumed_ = true;
    auto end  = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

} // namespace util
} // namespace langchain
