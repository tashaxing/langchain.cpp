// examples/24_multi_agent_workflow.cpp — First-phase Multi-Agent orchestration.
//
// Demonstrates CoordinatorAgent:
//   1. Register multiple agent runners as named AgentNode entries.
//   2. Execute them sequentially.
//   3. Pass previous outputs to downstream agents.
//   4. Aggregate all outputs into one result.
//
// No network required; uses deterministic in-process runners.
#include "langchain.h"

#include <iostream>
#include <string>

namespace
{

langchain::agent::AgentResult make_result(const std::string& output)
{
    langchain::agent::AgentResult r;
    r.output = output;
    r.finished = true;
    return r;
}

} // namespace

int main()
{
    using namespace langchain;

    agent::MultiAgentConfig cfg;
    cfg.summary_header = "Coordinator summary";
    cfg.stop_on_error = true;
    cfg.pass_previous_outputs = true;

    agent::CoordinatorAgent coordinator(cfg);

    // Researcher: extracts facts from the user request.
    coordinator.add_agent({
        "researcher",
        "Extracts relevant facts and constraints.",
        [](const std::string& input) -> agent::AgentResult
        {
            return make_result(
                "Facts:\n"
                "- The request asks for a C++ multi-agent workflow.\n"
                "- It should demonstrate orchestration without network calls.\n"
                "- The output should be easy to inspect.");
        }
    });

    // Planner: sees original request + researcher output.
    coordinator.add_agent({
        "planner",
        "Turns research notes into an implementation outline.",
        [](const std::string& input) -> agent::AgentResult
        {
            return make_result(
                "Plan:\n"
                "1. Define named agent nodes.\n"
                "2. Execute each node in sequence.\n"
                "3. Pass prior outputs to downstream nodes.\n"
                "4. Aggregate outputs into a final report.");
        }
    });

    // Writer: sees original request + previous outputs.
    coordinator.add_agent({
        "writer",
        "Writes the final answer from prior agent outputs.",
        [](const std::string& input) -> agent::AgentResult
        {
            return make_result(
                "Final answer:\n"
                "CoordinatorAgent provides a lightweight sequential multi-agent "
                "workflow. Each participant receives the original task plus prior "
                "outputs, then the coordinator aggregates the trace.");
        }
    });

    try
    {
        auto result = coordinator.invoke(
            "Explain the first-phase multi-agent orchestration design.");

        std::cout << "Finished: " << (result.finished ? "yes" : "no") << "\n";
        std::cout << "Steps: " << result.steps.size() << "\n\n";

        for (const auto& step : result.steps)
        {
            std::cout << "=== " << step.agent_name << " ===\n";
            std::cout << step.result.output << "\n\n";
        }

        std::cout << "=== Aggregated Output ===\n";
        std::cout << result.output << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "multi-agent workflow failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
