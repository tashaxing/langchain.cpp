// src/util/timer.cpp
#include "util/timer.h"
#include "util/logging.h"

namespace langchain
{
namespace util
{

Timer::Timer() = default;

Timer::~Timer()
{
    stop();
}

void Timer::start_once(std::chrono::milliseconds delay, Callback cb)
{
    stop();
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb_               = std::move(cb);
        interval_         = delay;
        recurring_        = false;
        fire_immediately_ = false;
        stop_flag_        = false;
        running_          = true;
    }
    worker_ = std::thread(&Timer::run, this);
}

void Timer::start_interval(std::chrono::milliseconds interval,
                           Callback cb,
                           bool fire_immediately)
{
    stop();
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb_               = std::move(cb);
        interval_         = interval;
        recurring_        = true;
        fire_immediately_ = fire_immediately;
        stop_flag_        = false;
        running_          = true;
    }
    worker_ = std::thread(&Timer::run, this);
}

void Timer::stop()
{
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!running_)
        {
            return;
        }
        stop_flag_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable())
    {
        worker_.join();
    }
    running_ = false;
}

bool Timer::is_running() const
{
    return running_.load();
}

void Timer::run()
{
    Callback cb_local;
    std::chrono::milliseconds interval;
    bool recurring;
    bool fire_now;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb_local  = cb_;
        interval  = interval_;
        recurring = recurring_;
        fire_now  = fire_immediately_;
    }

    if (recurring && fire_now)
    {
        // Snapshot stop flag without locking the cv path.
        if (!stop_flag_.load())
        {
            try
            {
                cb_local();
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("Timer callback threw: {}", e.what());
            }
            catch (...)
            {
                LOG_ERROR("Timer callback threw unknown exception");
            }
        }
    }

    while (true)
    {
        std::unique_lock<std::mutex> lk(mu_);
        if (cv_.wait_for(lk, interval, [&] { return stop_flag_.load(); }))
        {
            return; // stop requested
        }
        lk.unlock();

        try
        {
            cb_local();
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Timer interval callback threw: {}", e.what());
        }
        catch (...)
        {
            LOG_ERROR("Timer interval callback threw unknown exception");
        }

        if (!recurring)
        {
            return;
        }
    }
}

} // namespace util
} // namespace langchain
