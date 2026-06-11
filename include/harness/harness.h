// langchain/harness/harness.h
// Agent harness: self-check, self-feedback, self-correction, self-evolution.
//
// Wraps any agent runner (function, ReActAgent, ToolCallingAgent) and adds
// a 4-phase loop after each run:
//   1. Execute   → result + trace
//   2. Evaluate  → LLM scores output, gives critique
//   3. Correct   → if score < threshold, re-run with feedback injected
//   4. Evolve    → persist prompt refinements / strategy deltas to memory
//
// Usage:
//   HarnessConfig cfg;
//   cfg.min_score = 0.8f;
//   cfg.max_correction_iterations = 2;
//
//   HarnessAgent harness(cfg, llm);
//   harness.set_runner([&](const std::string& input) {
//       ReActAgent agent(llm, tools);
//       return agent.invoke(input);
//   });
//
//   auto result = harness.invoke("What is 17 * 23?");
#pragma once

#include "agent/agent.h"
#include "llm/llm.h"
#include "memory/memory.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace harness
{

// ---------------------------------------------------------------------------
// Result of a harness evaluation.
// ---------------------------------------------------------------------------
struct CheckResult
{
    float score = 0.0f;           // 0.0 .. 1.0
    std::string critique;         // detailed evaluation
    std::string suggestions;      // concrete improvements
    bool passed = false;          // score >= threshold
};

// ---------------------------------------------------------------------------
// Full trace of a harness run (original + corrections).
// ---------------------------------------------------------------------------
struct HarnessTrace
{
    std::string final_output;
    bool succeeded = false;       // true if final check passed or max iter reached
    int correction_iterations = 0;

    struct Iteration
    {
        int index = 0;            // 0 = original, 1+ = corrections
        agent::AgentResult agent_result;
        CheckResult check;
    };
    std::vector<Iteration> iterations;
};

// ---------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------
struct HarnessConfig
{
    float min_score = 0.75f;                  // threshold to pass evaluation
    int max_correction_iterations = 2;        // how many retry loops
    bool enable_evolution = true;             // persist learnings to memory
    std::string evaluation_criteria;          // custom rubric (optional)
};

// ---------------------------------------------------------------------------
// Runner abstraction — anything that turns a user string into an AgentResult.
// ---------------------------------------------------------------------------
using AgentRunner = std::function<agent::AgentResult(const std::string& user_input)>;

// ---------------------------------------------------------------------------
// HarnessAgent — wraps a runner with the 4-phase loop.
// ---------------------------------------------------------------------------
class HarnessAgent
{
public:
    HarnessAgent(HarnessConfig cfg,
                 llm::LLMPtr evaluator_llm,
                 memory::MemoryPtr mem = nullptr);

    // Set the agent to wrap. Must be called before invoke().
    void set_runner(AgentRunner runner);

    // Run the full harness loop.
    HarnessTrace invoke(const std::string& user_input);

    // Accessors.
    const HarnessConfig& config() const
    {
        return cfg_;
    }

    void set_hooks(hook::HookManager* mgr)
    {
        hooks_ = mgr;
    }

private:
    // Phase 2: evaluate output.
    CheckResult evaluate(const std::string& user_input,
                         const agent::AgentResult& result);

    // Phase 3: build a corrected input with feedback injected.
    std::string build_correction_prompt(const std::string& original_input,
                                        const CheckResult& check,
                                        const agent::AgentResult& prev_result);

    // Phase 4: persist learnings.
    void evolve(const HarnessTrace& trace);

    HarnessConfig cfg_;
    llm::LLMPtr llm_;
    memory::MemoryPtr mem_;
    AgentRunner runner_;
    hook::HookManager* hooks_ = nullptr;
};

} // namespace harness
} // namespace langchain
