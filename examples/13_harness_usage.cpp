// examples/13_harness_usage.cpp -- Harness (self-check / self-feedback / self-correct / self-evolve)
//
// Wraps a ReActAgent with HarnessAgent. The evaluator LLM scores each attempt
// and, if the score is below threshold, injects feedback so the agent can try
// again. Past critiques are persisted to memory so future runs learn from
// earlier mistakes.
//
// Run with env: LC_BASE_URL, LC_API_KEY, LC_MODEL.
#include "langchain.h"

#include <cstdlib>
#include <iostream>

int main()
{
    using namespace langchain;

    // ------------------------------------------------------------------
    // 1. Create the LLM (used both by the agent and the harness evaluator).
    // ------------------------------------------------------------------
    llm::OpenAILLMConfig cfg;
    if (const char* e = std::getenv("LC_BASE_URL"))
    {
        cfg.base_url = e;
    }
    if (const char* e = std::getenv("LC_API_KEY"))
    {
        cfg.api_key = e;
    }
    if (const char* e = std::getenv("LC_MODEL"))
    {
        cfg.model = e;
    }
    auto llm = std::make_shared<llm::OpenAILLM>(cfg);

    // ------------------------------------------------------------------
    // 2. Tools for the agent.
    // ------------------------------------------------------------------
    tool::ToolRegistry tools;
    tools.add(tool::make_calculator_tool());

    // ------------------------------------------------------------------
    // 3. Memory shared between agent and harness (evolution persists here).
    // ------------------------------------------------------------------
    auto mem = std::make_shared<memory::BufferMemory>();

    // ------------------------------------------------------------------
    // 4. Build the harness.
    // ------------------------------------------------------------------
    harness::HarnessConfig hcfg;
    hcfg.min_score = 0.8f;               // require 80 % to pass
    hcfg.max_correction_iterations = 2;  // up to 2 retries
    hcfg.enable_evolution = true;        // persist critiques to memory

    harness::HarnessAgent harness(hcfg, llm, mem);

    // The runner is a plain lambda; HarnessAgent can wrap any AgentRunner.
    harness.set_runner([&](const std::string& input) -> agent::AgentResult {
        agent::AgentConfig acfg;
        acfg.max_iterations = 6;
        agent::ReActAgent agent(llm, std::move(tools), acfg, mem);
        // Note: tools registry is moved in; rebuild for next iteration if needed.
        // In a real app you'd share the registry or clone it.
        return agent.invoke(input);
    });

    // ------------------------------------------------------------------
    // 5. Run a question that the agent might get wrong on the first try.
    // ------------------------------------------------------------------
    std::string question = "What is (17 * 23) + (8 / 2) - 3?";

    try
    {
        auto trace = harness.invoke(question);

        std::cout << "========== Harness Result ==========\n";
        std::cout << "Question : " << question << "\n";
        std::cout << "Final    : " << trace.final_output << "\n";
        std::cout << "Success  : " << (trace.succeeded ? "yes" : "no") << "\n";
        std::cout << "Iterations: " << (trace.correction_iterations + 1) << "\n\n";

        for (const auto& it : trace.iterations)
        {
            std::cout << "--- Iteration " << it.index << " ---\n";
            std::cout << "Score    : " << it.check.score << "\n";
            std::cout << "Passed   : " << (it.check.passed ? "yes" : "no") << "\n";
            std::cout << "Critique : " << it.check.critique << "\n";
            if (!it.check.suggestions.empty())
            {
                std::cout << "Suggestions: " << it.check.suggestions << "\n";
            }
            std::cout << "\n";
        }

        std::cout << "Memory now holds " << mem->messages().size()
                  << " messages (including learnings).\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "harness failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
