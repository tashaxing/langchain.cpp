// src/util/thread_pool.cpp — ThreadPool implementation.
#include "util/thread_pool.h"

#include "util/logging.h"

namespace langchain
{
namespace util
{

ThreadPool::ThreadPool(std::size_t num_threads)
{
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i)
    {
        workers_.emplace_back([this]
        {
            for (;;)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty())
                    {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                try
                {
                    task();
                }
                catch (const std::exception& e)
                {
                    LOG_ERROR("[ThreadPool] task threw: {}", e.what());
                }
                catch (...)
                {
                    LOG_ERROR("[ThreadPool] task threw unknown exception");
                }
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_)
    {
        if (w.joinable())
        {
            w.join();
        }
    }
}

bool ThreadPool::submit(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_)
        {
            return false;
        }
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
    return true;
}

ThreadPool& ThreadPool::default_pool()
{
    static ThreadPool pool(
        std::max<std::size_t>(2, std::thread::hardware_concurrency()));
    return pool;
}

} // namespace util
} // namespace langchain
