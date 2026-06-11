#include "agent/multi_agent.h"

#include <sstream>

namespace langchain
{
namespace agent
{

CoordinatorAgent::CoordinatorAgent(MultiAgentConfig cfg)
    : cfg_(std::move(cfg))
{
}

void CoordinatorAgent::add_agent(AgentNode node)
{
    if (node.name.empty())
    {
        throw LCError("CoordinatorAgent: agent name is required");
    }
    if (!node.runner)
    {
        throw LCError("CoordinatorAgent: runner is required for agent '" + node.name + "'");
    }
    agents_.push_back(std::move(node));
}

std::size_t CoordinatorAgent::size() const
{
    return agents_.size();
}

MultiAgentResult CoordinatorAgent::invoke(const std::string& user_input)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "CoordinatorAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_input;
    hook::ScopedSpan span(mgr, before, hook::Phase::AfterAgent);

    if (agents_.empty())
    {
        throw LCError("CoordinatorAgent: no agents registered");
    }

    MultiAgentResult out;
    for (const auto& node : agents_)
    {
        MultiAgentStep step;
        step.agent_name = node.name;
        step.input = build_input(user_input, out.steps);

        try
        {
            step.result = node.runner(step.input);
        }
        catch (const std::exception& e)
        {
            step.error = e.what();
            step.result.finished = false;
            step.result.output = "error: " + step.error;
            out.finished = false;
            out.steps.push_back(std::move(step));
            if (cfg_.stop_on_error)
            {
                break;
            }
            continue;
        }

        if (!step.result.finished)
        {
            out.finished = false;
            out.steps.push_back(std::move(step));
            if (cfg_.stop_on_error)
            {
                break;
            }
            continue;
        }

        out.steps.push_back(std::move(step));
    }

    out.output = aggregate(out.steps);
    span.after().agent_output = &out.output;
    return out;
}

AgentResult CoordinatorAgent::invoke_as_agent(const std::string& user_input)
{
    auto multi = invoke(user_input);
    AgentResult out;
    out.output = std::move(multi.output);
    out.finished = multi.finished;
    for (const auto& step : multi.steps)
    {
        AgentStep agent_step;
        agent_step.tool_name = step.agent_name;
        agent_step.tool_input = step.input;
        agent_step.observation = step.error.empty()
                                      ? step.result.output
                                      : "error: " + step.error;
        out.steps.push_back(std::move(agent_step));
    }
    return out;
}

std::string CoordinatorAgent::build_input(
    const std::string& original_input,
    const std::vector<MultiAgentStep>& steps) const
{
    if (!cfg_.pass_previous_outputs || steps.empty())
    {
        return original_input;
    }

    std::ostringstream oss;
    oss << "Original request:\n" << original_input << "\n\n"
        << "Previous agent outputs:\n";
    for (const auto& step : steps)
    {
        oss << "\n[" << step.agent_name << "]\n"
            << step.result.output << "\n";
    }
    oss << "\nUse the original request and previous outputs to produce your result.";
    return oss.str();
}

std::string CoordinatorAgent::aggregate(const std::vector<MultiAgentStep>& steps) const
{
    std::ostringstream oss;
    oss << cfg_.summary_header << "\n";
    for (const auto& step : steps)
    {
        oss << "\n## " << step.agent_name << "\n";
        if (!step.error.empty())
        {
            oss << "error: " << step.error << "\n";
        }
        else
        {
            oss << step.result.output << "\n";
        }
    }
    return oss.str();
}

} // namespace agent
} // namespace langchain
