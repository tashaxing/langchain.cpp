// langchain/util/thread_pool.h
// Lightweight fixed-size thread pool for background task execution.
//
// Used by the default async_invoke_impl to avoid spawning a new thread
// per LLM call.  The pool is created lazily via default_pool() so
// applications that never use the async API pay no thread cost.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace langchain
{
namespace util
{

class ThreadPool
{
public:
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueue a task.  Tasks are executed FIFO by the worker threads.
    // Returns true if the task was submitted, false if the pool is shutting down.
    bool submit(std::function<void()> task);

    // Process-wide default pool.  Created on first call with
    // std::thread::hardware_concurrency() workers (minimum 2).
    static ThreadPool& default_pool();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace util
} // namespace langchain
