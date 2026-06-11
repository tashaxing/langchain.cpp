// langchain/util/time.h
// Wall-clock + steady-clock helpers: ISO8601 formatting, millisecond
// timestamps, monotonic duration measurement. Cross-platform (Win/Linux/Mac).
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace langchain
{
namespace util
{

// Milliseconds since Unix epoch (wall clock).
std::int64_t now_ms();

// Seconds since process start (steady, monotonic).
double mono_seconds();

// "2026-05-26T07:42:13.123Z" — UTC, millisecond precision.
std::string iso8601_utc(std::int64_t epoch_ms);

// Same but using current time.
std::string iso8601_utc_now();

// Cheap scoped timer: prints elapsed ms to stderr on destruction unless
// the result was already consumed via `elapsed_ms()`. Intended for ad-hoc
// profiling, not production telemetry.
class ScopedTimer
{
public:
    explicit ScopedTimer(std::string label = {});
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    // Elapsed milliseconds since construction. Marks the timer as consumed
    // so the destructor will not print.
    double elapsed_ms();

private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
    bool consumed_{false};
};

} // namespace util
} // namespace langchain
