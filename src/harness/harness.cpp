// src/harness/harness.cpp -- Agent harness implementation.
#include "harness/harness.h"
#include "harness/reflection.h"

#include "hook/hook.h"
#include "util/strings.h"

#include <sstream>

namespace langchain
{
namespace harness
{

namespace
{

// Parse evaluator LLM response into CheckResult.
CheckResult parse_evaluation(const std::string& text)
{
    CheckResult out;
    out.score = 0.0f;
    out.passed = false;

    auto extract = [&](const std::string& key) -> std::string
    {
        auto pos = text.find(key);
        if (pos == std::string::npos)
        {
            return std::string();
        }
        pos += key.size();
        auto end = text.find('\n', pos);
        std::string v = text.substr(pos,
            end == std::string::npos ? std::string::npos : end - pos);
        return strings::trim(v);
    };

    std::string score_str = extract("SCORE:");
    if (!score_str.empty())
    {
        try
        {
            out.score = std::stof(score_str);
        }
        catch (...)
        {
            out.score = 0.0f;
        }
    }

    out.critique    = extract("CRITIQUE:");
    out.suggestions = extract("SUGGESTIONS:");

    // Fallback: if no structured fields, use the whole text as critique.
    if (out.critique.empty() && out.suggestions.empty())
    {
        out.critique = text;
    }

    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// HarnessAgent
// ---------------------------------------------------------------------------

HarnessAgent::HarnessAgent(HarnessConfig cfg,
                           llm::LLMPtr evaluator_llm,
                           memory::MemoryPtr mem)
    : cfg_(std::move(cfg)),
      llm_(std::move(evaluator_llm)),
      mem_(std::move(mem))
{
}

void HarnessAgent::set_runner(AgentRunner runner)
{
    runner_ = std::move(runner);
}

HarnessTrace HarnessAgent::invoke(const std::string& user_input)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "HarnessAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_input;
    hook::ScopedSpan span(mgr, before, hook::Phase::AfterAgent);

    HarnessTrace trace;
    std::string current_input = user_input;

    // Phase 1: initial run.
    agent::AgentResult result = runner_(current_input);

    HarnessTrace::Iteration iter0;
    iter0.index = 0;
    iter0.agent_result = result;
    iter0.check = evaluate(user_input, result);
    trace.iterations.push_back(std::move(iter0));

    // Phase 2+3: correction loop.
    while (!trace.iterations.back().check.passed &&
           trace.correction_iterations < cfg_.max_correction_iterations)
    {
        trace.correction_iterations++;
        const auto& last = trace.iterations.back();

        current_input = build_correction_prompt(user_input, last.check, last.agent_result);
        agent::AgentResult corrected = runner_(current_input);

        HarnessTrace::Iteration iter;
        iter.index = trace.correction_iterations;
        iter.agent_result = corrected;
        iter.check = evaluate(user_input, corrected);
        trace.iterations.push_back(std::move(iter));
    }

    // Pick the best result (highest score).
    const HarnessTrace::Iteration* best = &trace.iterations.front();
    for (const auto& it : trace.iterations)
    {
        if (it.check.score > best->check.score)
        {
            best = &it;
        }
    }

    trace.final_output = best->agent_result.output;
    trace.succeeded = best->check.passed;

    // Phase 4: evolve.
    if (cfg_.enable_evolution)
    {
        evolve(trace);
    }

    span.after().agent_output = &trace.final_output;
    return trace;
}

CheckResult HarnessAgent::evaluate(const std::string& user_input,
                                    const agent::AgentResult& result)
{
    std::string prompt = default_evaluation_prompt(user_input, result,
                                                      cfg_.evaluation_criteria);

    llm::ChatRequest req;
    req.messages.push_back(Message::user(prompt));
    req.temperature = 0.2f; // low temperature for consistent evaluation

    auto resp = llm_->invoke(req);
    CheckResult check = parse_evaluation(resp.message.content);
    check.passed = (check.score >= cfg_.min_score);
    return check;
}

std::string HarnessAgent::build_correction_prompt(const std::string& original_input,
                                                    const CheckResult& check,
                                                    const agent::AgentResult& prev)
{
    return default_correction_prompt(original_input, check, prev);
}

void HarnessAgent::evolve(const HarnessTrace& trace)
{
    if (!mem_)
    {
        return;
    }

    // Persist the critique history as a system message so future runs
    // can learn from past mistakes.
    std::ostringstream oss;
    oss << "[Harness Learnings] Previous attempts on similar questions:\n";
    for (const auto& it : trace.iterations)
    {
        if (it.index == 0)
        {
            oss << "- Attempt " << it.index << ": score=" << it.check.score
               << ", critique=" << it.check.critique << "\n";
        }
        else
        {
            oss << "- Correction " << it.index << ": score=" << it.check.score
               << ", critique=" << it.check.critique << "\n";
        }
    }

    Message learning_msg = Message::system(oss.str());
    mem_->add(learning_msg);
}

} // namespace harness
} // namespace langchain
