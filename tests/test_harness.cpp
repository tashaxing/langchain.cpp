// tests/test_harness.cpp — Harness unit tests.
//
// Covers HarnessAgent self-check loop, evaluation scoring, and correction.
#include <gtest/gtest.h>

#include "harness/harness.h"
#include "llm/llm.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// EvalLLM — returns a canned evaluation score.
// ---------------------------------------------------------------------------
class EvalLLM : public llm::ILLM
{
public:
    std::string name() const override { return "eval"; }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
    {
        llm::ChatResponse out;
        out.message = Message::assistant(
            "SCORE: 0.9\n"
            "CRITIQUE: good work\n"
            "SUGGESTIONS: none\n");
        out.finish_reason = "stop";
        return out;
    }
};

// ---------------------------------------------------------------------------
// LowScoreLLM — returns a low score on first evaluation, high on second.
// ---------------------------------------------------------------------------
class LowScoreLLM : public llm::ILLM
{
public:
    std::string name() const override { return "lowscore"; }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
    {
        llm::ChatResponse out;
        if (first_)
        {
            first_ = false;
            out.message = Message::assistant(
                "SCORE: 0.3\n"
                "CRITIQUE: bad\n"
                "SUGGESTIONS: try harder\n");
        }
        else
        {
            out.message = Message::assistant(
                "SCORE: 0.95\n"
                "CRITIQUE: great\n"
                "SUGGESTIONS: none\n");
        }
        out.finish_reason = "stop";
        return out;
    }

private:
    bool first_ = true;
};

// ---------------------------------------------------------------------------
// FixedRunner — deterministic agent result for testing.
// ---------------------------------------------------------------------------
agent::AgentResult make_runner_result(const std::string& output)
{
    agent::AgentResult r;
    r.output = output;
    r.finished = true;
    return r;
}

} // namespace

// ---------------------------------------------------------------------------
// HarnessAgent — basic invocation
// ---------------------------------------------------------------------------
TEST(HarnessAgent, InvokeReturnsTrace)
{
    auto eval_llm = std::make_shared<EvalLLM>();
    harness::HarnessConfig cfg;
    cfg.min_score = 0.5f;
    cfg.max_correction_iterations = 0; // no corrections
    harness::HarnessAgent harness(cfg, eval_llm);

    harness.set_runner([](const std::string&) -> agent::AgentResult
    {
        return make_runner_result("answer");
    });

    auto trace = harness.invoke("question");

    EXPECT_EQ(trace.final_output, "answer");
    EXPECT_TRUE(trace.succeeded);
    ASSERT_EQ(trace.iterations.size(), 1u);
    EXPECT_EQ(trace.iterations[0].check.score, 0.9f);
    EXPECT_TRUE(trace.iterations[0].check.passed);
}

// ---------------------------------------------------------------------------
// HarnessAgent — correction loop
// ---------------------------------------------------------------------------
TEST(HarnessAgent, CorrectionLoopRetries)
{
    auto eval_llm = std::make_shared<LowScoreLLM>();
    harness::HarnessConfig cfg;
    cfg.min_score = 0.8f;
    cfg.max_correction_iterations = 2;
    harness::HarnessAgent harness(cfg, eval_llm);

    int call_count = 0;
    harness.set_runner([&call_count](const std::string& input) -> agent::AgentResult
    {
        ++call_count;
        // Second call should receive the correction prompt.
        return make_runner_result("attempt " + std::to_string(call_count));
    });

    auto trace = harness.invoke("question");

    EXPECT_EQ(trace.final_output, "attempt 2");
    EXPECT_TRUE(trace.succeeded);
    EXPECT_EQ(trace.correction_iterations, 1);
    ASSERT_EQ(trace.iterations.size(), 2u);
    EXPECT_FALSE(trace.iterations[0].check.passed);
    EXPECT_TRUE(trace.iterations[1].check.passed);
}

// ---------------------------------------------------------------------------
// HarnessAgent — picks best result
// ---------------------------------------------------------------------------
TEST(HarnessAgent, PicksBestResult)
{
    auto eval_llm = std::make_shared<EvalLLM>();
    harness::HarnessConfig cfg;
    cfg.min_score = 1.0f; // impossible threshold
    cfg.max_correction_iterations = 1;
    harness::HarnessAgent harness(cfg, eval_llm);

    harness.set_runner([](const std::string&) -> agent::AgentResult
    {
        static int count = 0;
        ++count;
        return make_runner_result("v" + std::to_string(count));
    });

    auto trace = harness.invoke("q");

    // Both iterations score 0.9 (below 1.0), so best is the higher score.
    // Since EvalLLM always returns 0.9, both are equal; best = first.
    EXPECT_EQ(trace.final_output, "v1");
    EXPECT_FALSE(trace.succeeded);
}

// ---------------------------------------------------------------------------
// HarnessAgent — evolution persists learnings
// ---------------------------------------------------------------------------
TEST(HarnessAgent, EvolutionPersistsToMemory)
{
    auto eval_llm = std::make_shared<EvalLLM>();
    auto mem = std::make_shared<memory::BufferMemory>();

    harness::HarnessConfig cfg;
    cfg.min_score = 0.5f;
    cfg.max_correction_iterations = 0;
    cfg.enable_evolution = true;
    harness::HarnessAgent harness(cfg, eval_llm, mem);

    harness.set_runner([](const std::string&) -> agent::AgentResult
    {
        return make_runner_result("result");
    });

    harness.invoke("q");

    auto msgs = mem->messages();
    EXPECT_FALSE(msgs.empty());
    EXPECT_EQ(msgs.back().role, Role::System);
    EXPECT_NE(msgs.back().content.find("Harness Learnings"), std::string::npos);
}

// ---------------------------------------------------------------------------
// HarnessAgent — null memory does not crash
// ---------------------------------------------------------------------------
TEST(HarnessAgent, NullMemorySafe)
{
    auto eval_llm = std::make_shared<EvalLLM>();
    harness::HarnessConfig cfg;
    cfg.enable_evolution = true; // evolution on but no memory
    harness::HarnessAgent harness(cfg, eval_llm, nullptr);

    harness.set_runner([](const std::string&) -> agent::AgentResult
    {
        return make_runner_result("ok");
    });

    auto trace = harness.invoke("q");
    EXPECT_EQ(trace.final_output, "ok");
}
