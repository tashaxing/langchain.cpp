// examples/17_hooks_observability.cpp — HookManager lifecycle observability.
//
// Demonstrates:
//   1. Lambda-based hooks (FunctionHook) filtering by phase.
//   2. Custom IHook subclass for structured logging.
//   3. ScopedSpan for automatic Before/After timing.
//   4. Metadata sharing across phases.
//
// No network required — uses a mock EchoLLM.
#include "langchain.h"

#include <iostream>
#include <sstream>

// ---------------------------------------------------------------------------
// Custom hook: counts events per phase and prints a summary.
// ---------------------------------------------------------------------------
class StatsHook : public langchain::hook::IHook
{
public:
    std::string name() const override
    {
        return "stats";
    }

    void on_event(langchain::hook::HookContext& ctx) override
    {
        ++counts_[static_cast<int>(ctx.phase)];
        if (ctx.phase == langchain::hook::Phase::AfterLLM && ctx.elapsed.count() > 0)
        {
            total_llm_us_ += ctx.elapsed.count();
        }
    }

    void print_summary() const
    {
        std::cout << "\n=== Hook Stats ===\n";
        for (const auto& kv : counts_)
        {
            std::cout << "  " << langchain::hook::to_string(
                static_cast<langchain::hook::Phase>(kv.first))
                      << ": " << kv.second << "\n";
        }
        if (total_llm_us_ > 0)
        {
            std::cout << "  Total LLM time: " << total_llm_us_ << " us\n";
        }
    }

private:
    mutable std::map<int, int> counts_;
    mutable std::int64_t total_llm_us_ = 0;
};

// ---------------------------------------------------------------------------
// Mock LLM for deterministic demo.
// ---------------------------------------------------------------------------
class EchoLLM : public langchain::llm::ILLM
{
public:
    std::string name() const override
    {
        return "echo";
    }

protected:
    langchain::llm::ChatResponse invoke_impl(
        const langchain::llm::ChatRequest& req) override
    {
        langchain::llm::ChatResponse out;
        std::string last;
        for (const auto& m : req.messages)
        {
            if (m.role == langchain::Role::User)
            {
                last = m.content;
            }
        }
        out.message = langchain::Message::assistant("echo: " + last);
        out.finish_reason = "stop";
        return out;
    }
};

int main()
{
    using namespace langchain;

    // ---- 1. Lambda hook filtered to LLM phases only ----
    hook::HookManager mgr;
    mgr.add("llm-tracer",
        [](hook::HookContext& ctx)
    {
        std::cout << "[llm-tracer] " << hook::to_string(ctx.phase)
                  << " component=" << ctx.component
                  << " call_id=" << ctx.call_id;
        if (ctx.elapsed.count() > 0)
        {
            std::cout << " elapsed=" << ctx.elapsed.count() << "us";
        }
        std::cout << "\n";
    },
        {hook::Phase::BeforeLLM, hook::Phase::AfterLLM});

    // ---- 2. Custom stats hook (all phases) ----
    auto stats = std::make_shared<StatsHook>();
    mgr.add(stats);

    // ---- 3. Metadata-sharing hook ----
    mgr.add("metadata-tag",
        [](hook::HookContext& ctx)
    {
        if (ctx.phase == hook::Phase::BeforeLLM)
        {
            ctx.metadata["start_time"] =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
        }
        else if (ctx.phase == hook::Phase::AfterLLM)
        {
            auto it = ctx.metadata.find("start_time");
            if (it != ctx.metadata.end())
            {
                std::cout << "[metadata-tag] LLM call tagged with start_time="
                          << it->get<std::int64_t>() << "\n";
            }
        }
    });

    // ---- 4. Attach hooks to LLM and invoke ----
    EchoLLM llm;
    llm.set_hooks(&mgr);

    std::cout << "=== Invoking LLM ===\n";
    llm::ChatRequest req;
    req.messages.push_back(Message::user("hello hooks"));
    auto resp = llm.invoke(req);
    std::cout << "Response: " << resp.message.content << "\n";

    // ---- 5. Agent with hooks ----
    std::cout << "\n=== Invoking Agent ===\n";
    tool::ToolRegistry tools;
    tools.add(tool::make_calculator_tool());

    agent::AgentConfig acfg;
    acfg.max_iterations = 3;
    agent::ReActAgent ag(
        std::make_shared<EchoLLM>(), std::move(tools), acfg);
    ag.set_hooks(&mgr);

    auto result = ag.invoke("What is 2+2?");
    std::cout << "Agent output: " << result.output << "\n";

    // ---- 6. Print stats summary ----
    stats->print_summary();

    return 0;
}
