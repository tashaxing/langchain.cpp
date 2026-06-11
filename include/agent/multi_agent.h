// langchain/agent/multi_agent.h
// First-phase multi-agent orchestration: a lightweight CoordinatorAgent that
// executes a sequence of agent runners and aggregates their outputs.
#pragma once

#include "agent/agent.h"
#include "harness/harness.h"
#include "hook/hook.h"
#include "util/common.h"

#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace agent
{

// One participant in a coordinator workflow.
struct AgentNode
{
    std::string name;
    std::string description;
    harness::AgentRunner runner;
};

struct MultiAgentConfig
{
    // Stop at the first agent that returns finished=false or throws.
    bool stop_on_error = true;

    // If true, each downstream agent sees the original input plus prior outputs.
    // If false, every agent receives the original input unchanged.
    bool pass_previous_outputs = true;

    // Header used when building the final aggregated answer.
    std::string summary_header = "Multi-agent result";
};

struct MultiAgentStep
{
    std::string agent_name;
    std::string input;
    AgentResult result;
    std::string error;
};

struct MultiAgentResult
{
    std::string output;
    std::vector<MultiAgentStep> steps;
    bool finished = true;
};

class CoordinatorAgent
{
public:
    explicit CoordinatorAgent(MultiAgentConfig cfg = {});

    void add_agent(AgentNode node);
    std::size_t size() const;

    MultiAgentResult invoke(const std::string& user_input);

    // Convenience adapter for APIs that expect a plain AgentResult.
    AgentResult invoke_as_agent(const std::string& user_input);

    void set_hooks(hook::HookManager* mgr)
    {
        hooks_ = mgr;
    }

private:
    std::string build_input(const std::string& original_input,
                            const std::vector<MultiAgentStep>& steps) const;
    std::string aggregate(const std::vector<MultiAgentStep>& steps) const;

    MultiAgentConfig cfg_;
    std::vector<AgentNode> agents_;
    hook::HookManager* hooks_ = nullptr;
};

} // namespace agent
} // namespace langchain
