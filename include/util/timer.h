// langchain/util/timer.h
// Lightweight timer running on a dedicated background thread. Supports both
// one-shot delay and recurring interval modes. Stop is safe + idempotent.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace langchain
{
namespace util
{

class Timer
{
public:
    using Callback = std::function<void()>;

    Timer();
    ~Timer();

    Timer(const Timer&)            = delete;
    Timer& operator=(const Timer&) = delete;

    // Fire `cb` exactly once after `delay`. Replaces any pending schedule.
    void start_once(std::chrono::milliseconds delay, Callback cb);

    // Fire `cb` every `interval` until stop() / destruction.
    // If `fire_immediately` is true, runs `cb` once before sleeping.
    void start_interval(std::chrono::milliseconds interval,
                        Callback cb,
                        bool fire_immediately = false);

    // Stop any pending firing. Idempotent. Blocks until the worker is joined.
    void stop();

    bool is_running() const;

private:
    void run();

    std::thread             worker_;
    std::mutex              mu_;
    std::condition_variable cv_;
    Callback                cb_;
    std::chrono::milliseconds interval_{0};
    bool                    recurring_{false};
    bool                    fire_immediately_{false};
    std::atomic<bool>       running_{false};
    std::atomic<bool>       stop_flag_{false};
};

} // namespace util
} // namespace langchain
