// examples/18_async_llm.cpp — Asynchronous LLM invocation.
//
// Demonstrates:
//   1. async_invoke: fire LLM call on background thread, callback on completion.
//   2. async_invoke_stream: stream deltas on background thread.
//   3. Hook firing: BeforeLLM on caller thread, AfterLLM on background thread.
//   4. Error propagation: exceptions caught and delivered as error strings.
//
// No network required — uses a mock EchoLLM.
#include "langchain.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>

// ---------------------------------------------------------------------------
// Mock LLM with a small delay so async behavior is visible.
// ---------------------------------------------------------------------------
class SlowEchoLLM : public langchain::llm::ILLM
{
public:
    std::string name() const override
    {
        return "slow-echo";
    }

protected:
    langchain::llm::ChatResponse invoke_impl(
        const langchain::llm::ChatRequest& req) override
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        langchain::llm::ChatResponse out;
        std::string last;
        for (const auto& m : req.messages)
        {
            if (m.role == langchain::Role::User)
            {
                last = m.content;
            }
        }
        out.message = langchain::Message::assistant("async echo: " + last);
        out.finish_reason = "stop";
        return out;
    }
};

// ---------------------------------------------------------------------------
// BoomLLM — always throws, for error-path demo.
// ---------------------------------------------------------------------------
class BoomLLM : public langchain::llm::ILLM
{
public:
    std::string name() const override
    {
        return "boom";
    }

protected:
    langchain::llm::ChatResponse invoke_impl(
        const langchain::llm::ChatRequest&) override
    {
        throw std::runtime_error("intentional async failure");
    }
};

int main()
{
    using namespace langchain;

    // ---- 1. async_invoke — basic success ----
    std::cout << "=== async_invoke (success) ===\n";
    {
        SlowEchoLLM llm;
        std::atomic<bool> done{false};
        std::string result;
        std::mutex mu;
        std::condition_variable cv;

        llm::ChatRequest req;
        req.messages.push_back(Message::user("hello async"));

        llm.async_invoke(req,
            [&done, &result, &mu, &cv
            ](const llm::ChatResponse* resp, const std::string* err)
        {
            std::lock_guard<std::mutex> lk(mu);
            if (resp)
            {
                result = resp->message.content;
            }
            else if (err)
            {
                result = "error: " + *err;
            }
            done.store(true);
            cv.notify_one();
        });

        std::cout << "  Request fired, waiting for callback...\n";
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&done] { return done.load(); });
        std::cout << "  Result: " << result << "\n";
    }

    // ---- 2. async_invoke — error path ----
    std::cout << "\n=== async_invoke (error) ===\n";
    {
        BoomLLM llm;
        std::atomic<bool> done{false};
        std::string err_msg;

        llm::ChatRequest req;
        req.messages.push_back(Message::user("boom"));

        llm.async_invoke(req,
            [&done, &err_msg
            ](const llm::ChatResponse* resp, const std::string* err)
        {
            if (err)
            {
                err_msg = *err;
            }
            done.store(true);
        });

        while (!done.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "  Error captured: " << err_msg << "\n";
    }

    // ---- 3. async_invoke_stream — deltas on background thread ----
    std::cout << "\n=== async_invoke_stream ===\n";
    {
        SlowEchoLLM llm;
        std::atomic<bool> done{false};
        std::string accumulated;
        std::mutex mu;

        llm::ChatRequest req;
        req.messages.push_back(Message::user("stream me"));

        llm.async_invoke_stream(req,
            [&accumulated, &mu
            ](const std::string& delta) -> bool
        {
            std::lock_guard<std::mutex> lk(mu);
            accumulated += delta;
            std::cout << "  [delta] " << delta << "\n";
            return true;
        },
            [&done
            ](const llm::ChatResponse* resp, const std::string* err)
        {
            done.store(true);
        });

        while (!done.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "  Accumulated: " << accumulated << "\n";
    }

    // ---- 4. Hooks fire on correct threads ----
    std::cout << "\n=== Hooks thread verification ===\n";
    {
        hook::HookManager mgr;
        std::thread::id before_thread;
        std::thread::id after_thread;
        std::atomic<bool> done{false};

        mgr.add("thread-check",
            [&before_thread, &after_thread
            ](hook::HookContext& ctx)
        {
            if (ctx.phase == hook::Phase::BeforeLLM)
            {
                before_thread = std::this_thread::get_id();
            }
            else if (ctx.phase == hook::Phase::AfterLLM)
            {
                after_thread = std::this_thread::get_id();
            }
        });

        SlowEchoLLM llm;
        llm.set_hooks(&mgr);

        llm::ChatRequest req;
        req.messages.push_back(Message::user("thread test"));
        llm.async_invoke(req,
            [&done
            ](const llm::ChatResponse*, const std::string*)
        {
            done.store(true);
        });

        while (!done.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "  BeforeLLM thread: " << before_thread << "\n";
        std::cout << "  AfterLLM thread:  " << after_thread << "\n";
        std::cout << "  Same thread? " << (before_thread == after_thread ? "yes" : "no")
                  << " (should be NO)\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
