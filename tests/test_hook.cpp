// tests/test_gtest_hook.cpp — lifecycle hook coverage.
#include <gtest/gtest.h>

#include "hook/hook.h"
#include "llm/llm.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace langchain;

namespace
{

// Minimal in-process LLM: echoes the last user message. Doesn't go over the
// network, so we can test hook semantics deterministically.
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
};

} // namespace

TEST(Hook, ManagerFiresInOrder)
{
    hook::HookManager mgr;
    std::vector<hook::Phase> phases;

    mgr.add("collector", [&](hook::HookContext& ctx)
    {
        phases.push_back(ctx.phase);
    });

    EchoLLM llm;
    llm.set_hooks(&mgr);

    llm::ChatRequest req;
    req.messages.push_back(Message::user("hi"));
    auto resp = llm.invoke(req);

    ASSERT_EQ(phases.size(), 2u);
    EXPECT_EQ(phases[0], hook::Phase::BeforeLLM);
    EXPECT_EQ(phases[1], hook::Phase::AfterLLM);
    EXPECT_EQ(resp.message.content, "echo: hi");
}

TEST(Hook, MutableRequestCanBeRewritten)
{
    hook::HookManager mgr;
    mgr.add("inject-system",
        [](hook::HookContext& ctx)
        {
            if (ctx.mutable_request)
            {
                ctx.mutable_request->messages.insert(
                    ctx.mutable_request->messages.begin(),
                    Message::system("you are tested"));
            }
        },
        { hook::Phase::BeforeLLM });

    // A bespoke LLM that records what it saw.
    class Recording : public llm::ILLM
    {
    public:
        std::vector<Message> seen;

        std::string name() const override
        {
            return "recording";
        }

    protected:
        llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
        {
            seen = req.messages;
            llm::ChatResponse r;
            r.message = Message::assistant("ok");
            return r;
        }
    };

    Recording llm;
    llm.set_hooks(&mgr);

    llm::ChatRequest req;
    req.messages.push_back(Message::user("hi"));
    llm.invoke(req);

    ASSERT_EQ(llm.seen.size(), 2u);
    EXPECT_EQ(llm.seen[0].role, Role::System);
    EXPECT_EQ(llm.seen[0].content, "you are tested");
    EXPECT_EQ(llm.seen[1].role, Role::User);
}

TEST(Hook, AfterPhaseElapsedIsPositive)
{
    hook::HookManager mgr;
    std::atomic<std::int64_t> elapsed_us{0};

    mgr.add("timer",
        [&](hook::HookContext& ctx)
        {
            elapsed_us.store(ctx.elapsed.count());
        },
        { hook::Phase::AfterLLM });

    class Slow : public llm::ILLM
    {
    public:
        std::string name() const override { return "slow"; }
    protected:
        llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            return {};
        }
    };

    Slow llm;
    llm.set_hooks(&mgr);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("x"));
    llm.invoke(req);

    EXPECT_GT(elapsed_us.load(), 0);
}

TEST(Hook, ExceptionInHookDoesNotAbortCall)
{
    hook::HookManager mgr;
    int after_hits = 0;

    mgr.add("boom", [](hook::HookContext&)
    {
        throw std::runtime_error("intentional");
    });
    mgr.add("counter",
        [&](hook::HookContext&) { ++after_hits; },
        { hook::Phase::AfterLLM });

    EchoLLM llm;
    llm.set_hooks(&mgr);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("hi"));
    auto resp = llm.invoke(req);

    EXPECT_EQ(resp.message.content, "echo: hi");
    EXPECT_EQ(after_hits, 1);
}

TEST(Hook, WantsFilterRespected)
{
    hook::HookManager mgr;
    int before_count = 0;
    int after_count = 0;

    mgr.add("only-before",
        [&](hook::HookContext&) { ++before_count; },
        { hook::Phase::BeforeLLM });
    mgr.add("only-after",
        [&](hook::HookContext&) { ++after_count; },
        { hook::Phase::AfterLLM });

    EchoLLM llm;
    llm.set_hooks(&mgr);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("hi"));
    llm.invoke(req);
    llm.invoke(req);

    EXPECT_EQ(before_count, 2);
    EXPECT_EQ(after_count, 2);
}

TEST(Hook, RemoveAndClear)
{
    hook::HookManager mgr;
    mgr.add("a", [](hook::HookContext&) {});
    mgr.add("b", [](hook::HookContext&) {});
    EXPECT_EQ(mgr.size(), 2u);
    EXPECT_TRUE(mgr.remove("a"));
    EXPECT_EQ(mgr.size(), 1u);
    EXPECT_FALSE(mgr.remove("nope"));
    mgr.clear();
    EXPECT_EQ(mgr.size(), 0u);
}

TEST(Hook, CallIdMatchesBeforeAndAfter)
{
    hook::HookManager mgr;
    std::string before_id;
    std::string after_id;

    mgr.add("watch", [&](hook::HookContext& ctx)
    {
        if (ctx.phase == hook::Phase::BeforeLLM)
        {
            before_id = ctx.call_id;
        }
        else if (ctx.phase == hook::Phase::AfterLLM)
        {
            after_id = ctx.call_id;
        }
    });

    EchoLLM llm;
    llm.set_hooks(&mgr);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("ping"));
    llm.invoke(req);

    EXPECT_FALSE(before_id.empty());
    EXPECT_EQ(before_id, after_id);
}
