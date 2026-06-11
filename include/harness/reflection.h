// langchain/harness/reflection.h
// Prompt templates and formatting helpers for the harness reflection loop.
#pragma once

#include "agent/agent.h"
#include "harness/harness.h"

#include <sstream>
#include <string>

namespace langchain
{
namespace harness
{

// Format an AgentResult + its steps into a concise string for the evaluator.
inline std::string format_trace(const agent::AgentResult& r)
{
    std::ostringstream oss;
    oss << "=== Agent Output ===\n"
       << r.output << "\n\n"
       << "=== Steps (" << r.steps.size() << ") ===\n";
    for (std::size_t i = 0; i < r.steps.size(); ++i)
    {
        const auto& s = r.steps[i];
        oss << "[" << (i + 1) << "] Tool: " << s.tool_name << "\n";
        if (!s.thought.empty())
        {
            oss << "    Thought: " << s.thought << "\n";
        }
        oss << "    Input:  " << s.tool_input << "\n";
        oss << "    Output: " << s.observation << "\n";
    }
    return oss.str();
}

// Default evaluation prompt. The LLM is asked to score and critique.
inline std::string default_evaluation_prompt(const std::string& user_input,
                                              const agent::AgentResult& result,
                                              const std::string& criteria)
{
    std::ostringstream oss;
    oss << "You are a strict evaluator. Review the agent's work and assign a score.\n\n"
       << "User Question:\n" << user_input << "\n\n"
       << format_trace(result) << "\n";

    if (!criteria.empty())
    {
        oss << "Evaluation Criteria:\n" << criteria << "\n\n";
    }
    else
    {
        oss << "Evaluate on:\n"
           << "- Correctness: is the final answer accurate?\n"
           << "- Reasoning:   are the tool calls well-chosen and logical?\n"
           << "- Completeness: did the agent finish or stop prematurely?\n\n";
    }

    oss << "Respond in this exact format (no markdown):\n"
       << "SCORE: <number between 0.0 and 1.0>\n"
       << "CRITIQUE: <one or two sentences>\n"
       << "SUGGESTIONS: <what the agent should do differently>\n";
    return oss.str();
}

// Correction prompt — injects feedback into the next agent run.
inline std::string default_correction_prompt(const std::string& original_input,
                                              const CheckResult& check,
                                              const agent::AgentResult& prev)
{
    std::ostringstream oss;
    oss << "The previous attempt to answer the following question was insufficient.\n\n"
       << "Question: " << original_input << "\n\n"
       << "Previous Answer: " << prev.output << "\n\n"
       << "Evaluation Score: " << check.score << "/1.0\n"
       << "Critique: " << check.critique << "\n"
       << "Suggestions: " << check.suggestions << "\n\n"
       << "Please try again, incorporating the suggestions above.\n";
    return oss.str();
}

} // namespace harness
} // namespace langchain
