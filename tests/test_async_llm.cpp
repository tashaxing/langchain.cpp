// tests/test_async_llm.cpp — async ILLM API coverage.
#include <gtest/gtest.h>

#include "llm/llm.h"
#include "hook/hook.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// EchoLLM — minimal in-process backend for deterministic testing.
// ---------------------------------------------------------------------------
class EchoLLM : public llm::ILLM
{
public:
    std::string name() const override
    {
        return "echo";
    }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
    {
        llm::ChatResponse out;
        out.model = "echo";
        std::string last;
        for (const auto& m : req.messages)
        {
            if (m.role == Role::User)
            {
                last = m.content;
            }
        }
        out.message = Message::assistant("echo: " + last);
        out.finish_reason = "stop";
        return out;
    }

    llm::ChatResponse invoke_stream_impl(const llm::ChatRequest& req,
                                          const llm::StreamCallback& on_delta) override
    {
        auto resp = invoke_impl(req);
        // Stream the content character-by-character to exercise the path.
        for (char c : resp.message.content)
        {
            if (!on_delta(std::string(1, c)))
            {
                break;
            }
        }
        return resp;
    }
};

// ---------------------------------------------------------------------------
// SlowLLM — sleeps briefly so we can verify background execution.
// ---------------------------------------------------------------------------
class SlowLLM : public llm::ILLM
{
public:
    std::string name() const override
    {
        return "slow";
    }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        llm::ChatResponse out;
        out.model = "slow";
        out.message = Message::assistant("done");
        out.finish_reason = "stop";
        return out;
    }
};

// ---------------------------------------------------------------------------
// BoomLLM — always throws.
// ---------------------------------------------------------------------------
class BoomLLM : public llm::ILLM
{
public:
    std::string name() const override
    {
        return "boom";
    }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
    {
        throw std::runtime_error("intentional failure");
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
bool wait_for(std::atomic<T>& value, T expected, std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (value.load() == expected)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

} // namespace

// ---------------------------------------------------------------------------
// async_invoke — basic success path
// ---------------------------------------------------------------------------
TEST(AsyncLLM, InvokeCompletesOnBackgroundThread)
{
    EchoLLM llm;

    std::atomic<bool> done{false};
    std::atomic<std::thread::id> callback_thread_id;
    std::thread::id caller_thread_id = std::this_thread::get_id();

    llm::ChatRequest req;
    req.messages.push_back(Message::user("hello"));

    llm.async_invoke(req,
        [&done, &callback_thread_id, caller_thread_id
        ](const llm::ChatResponse* resp, const std::string* err)
    {
        callback_thread_id.store(std::this_thread::get_id());
        EXPECT_NE(callback_thread_id.load(), caller_thread_id);
        EXPECT_NE(resp, nullptr);
        EXPECT_EQ(err, nullptr);
        EXPECT_EQ(resp->message.content, "echo: hello");
        done.store(true);
    });

    EXPECT_TRUE(wait_for(done, true, std::chrono::milliseconds(2000)));
}

// ---------------------------------------------------------------------------
// async_invoke — error propagation
// ---------------------------------------------------------------------------
TEST(AsyncLLM, InvokePropagatesExceptionAsError)
{
    BoomLLM llm;

    std::atomic<bool> done{false};
    llm::ChatRequest req;
    req.messages.push_back(Message::user("x"));

    llm.async_invoke(req,
        [&done](const llm::ChatResponse* resp, const std::string* err)
    {
        EXPECT_EQ(resp, nullptr);
        EXPECT_NE(err, nullptr);
        EXPECT_NE(err->find("intentional failure"), std::string::npos);
        done.store(true);
    });

    EXPECT_TRUE(wait_for(done, true, std::chrono::milliseconds(2000)));
}

// ---------------------------------------------------------------------------
// async_invoke_stream — success path
// ---------------------------------------------------------------------------
TEST(AsyncLLM, InvokeStreamDeliversDeltasAndCompletion)
{
    EchoLLM llm;

    std::atomic<bool> done{false};
    std::string accumulated;
    std::mutex mu;
    std::condition_variable cv;

    llm::ChatRequest req;
    req.messages.push_back(Message::user("hi"));

    llm.async_invoke_stream(req,
        [&accumulated, &mu
        ](const std::string& delta) -> bool
    {
        std::lock_guard<std::mutex> lk(mu);
        accumulated += delta;
        return true;
    },
        [&done, &cv
        ](const llm::ChatResponse* resp, const std::string* err)
    {
        EXPECT_NE(resp, nullptr);
        EXPECT_EQ(err, nullptr);
        EXPECT_EQ(resp->message.content, "echo: hi");
        done.store(true);
        cv.notify_one();
    });

    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::milliseconds(2000),
                [&done] { return done.load(); });
    EXPECT_TRUE(done.load());
    EXPECT_EQ(accumulated, "echo: hi");
}

// ---------------------------------------------------------------------------
// async_invoke — hooks fire Before on caller, After on background
// ---------------------------------------------------------------------------
TEST(AsyncLLM, HooksFireBeforeOnCallerAndAfterOnBackground)
{
    hook::HookManager mgr;
    std::atomic<std::thread::id> before_thread_id;
    std::atomic<std::thread::id> after_thread_id;
    std::atomic<bool> done{false};

    mgr.add("tracer",
        [&before_thread_id, &after_thread_id
        ](hook::HookContext& ctx)
    {
        if (ctx.phase == hook::Phase::BeforeLLM)
        {
            before_thread_id.store(std::this_thread::get_id());
        }
        else if (ctx.phase == hook::Phase::AfterLLM)
        {
            after_thread_id.store(std::this_thread::get_id());
        }
    });

    SlowLLM llm;
    llm.set_hooks(&mgr);

    std::thread::id caller_id = std::this_thread::get_id();

    llm::ChatRequest req;
    req.messages.push_back(Message::user("z"));

    llm.async_invoke(req,
        [&done](const llm::ChatResponse*, const std::string*)
    {
        done.store(true);
    });

    // Before should have fired immediately on the caller thread.
    EXPECT_EQ(before_thread_id.load(), caller_id);

    // Wait for completion.
    EXPECT_TRUE(wait_for(done, true, std::chrono::milliseconds(2000)));

    // After should have fired on a different (background) thread.
    EXPECT_NE(after_thread_id.load(), std::thread::id());
    EXPECT_NE(after_thread_id.load(), caller_id);
}

// ---------------------------------------------------------------------------
// async_invoke — call_id matches Before and After
// ---------------------------------------------------------------------------
TEST(AsyncLLM, HookCallIdMatchesBeforeAndAfter)
{
    hook::HookManager mgr;
    std::string before_id;
    std::string after_id;
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> done{false};

    mgr.add("matcher",
        [&before_id, &after_id, &mu, &cv, &done
        ](hook::HookContext& ctx)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (ctx.phase == hook::Phase::BeforeLLM)
        {
            before_id = ctx.call_id;
        }
        else if (ctx.phase == hook::Phase::AfterLLM)
        {
            after_id = ctx.call_id;
            done.store(true);
            cv.notify_one();
        }
    });

    EchoLLM llm;
    llm.set_hooks(&mgr);

    llm::ChatRequest req;
    req.messages.push_back(Message::user("ping"));

    llm.async_invoke(req, nullptr);

    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::milliseconds(2000),
                [&done] { return done.load(); });

    EXPECT_FALSE(before_id.empty());
    EXPECT_EQ(before_id, after_id);
}

// ---------------------------------------------------------------------------
// async_invoke — error path still fires After hook
// ---------------------------------------------------------------------------
TEST(AsyncLLM, AfterHookFiresEvenOnException)
{
    hook::HookManager mgr;
    std::atomic<int> after_count{0};
    std::atomic<bool> done{false};

    mgr.add("counter",
        [&after_count
        ](hook::HookContext& ctx)
    {
        if (ctx.phase == hook::Phase::AfterLLM)
        {
            ++after_count;
        }
    });

    BoomLLM llm;
    llm.set_hooks(&mgr);

    llm::ChatRequest req;
    req.messages.push_back(Message::user("boom"));

    llm.async_invoke(req,
        [&done](const llm::ChatResponse*, const std::string*)
    {
        done.store(true);
    });

    EXPECT_TRUE(wait_for(done, true, std::chrono::milliseconds(2000)));
    EXPECT_EQ(after_count.load(), 1);
}

// ---------------------------------------------------------------------------
// async_invoke — null callback is safe
// ---------------------------------------------------------------------------
TEST(AsyncLLM, NullCallbackIsSafe)
{
    EchoLLM llm;
    llm::ChatRequest req;
    req.messages.push_back(Message::user("test"));

    // Should not crash even with no callback.
    llm.async_invoke(req, nullptr);

    // Give the thread pool a moment to process.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
