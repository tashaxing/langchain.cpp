// examples/03_react_agent.cpp -- ReAct agent with the built-in calculator tool
// and persistent memory (SQLite). The agent remembers context across multiple
// runs within the same process.
//
// Run with env: LC_BASE_URL, LC_API_KEY, LC_MODEL.
#include "langchain.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main()
{
    using namespace langchain;

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

    tool::ToolRegistry tools;
    tools.add(tool::make_calculator_tool());

    // ---- Persistent SQLite memory ----
    // The agent will remember previous exchanges across runs in this session.
    std::filesystem::create_directories("build");
    auto mem = std::make_shared<memory::LongTermMemory>(
        memory::LongTermMemory::sqlite("build/react_agent_memory.db", "demo-session"));

    agent::AgentConfig acfg;
    acfg.max_iterations = 6;
    agent::ReActAgent ag(llm, std::move(tools), acfg, mem);

    try
    {
        auto r1 = ag.invoke("What is (17 * 23) + 9?");
        std::cout << "Run 1\n";
        std::cout << "  Answer: " << r1.output << "\n";
        std::cout << "  Steps : " << r1.steps.size()
                  << (r1.finished ? " (finished)" : " (max-iter)") << "\n";

        // Because memory is persistent, the agent knows what "that result" refers to.
        auto r2 = ag.invoke("Now multiply that result by 2.");
        std::cout << "Run 2\n";
        std::cout << "  Answer: " << r2.output << "\n";
        std::cout << "  Steps : " << r2.steps.size()
                  << (r2.finished ? " (finished)" : " (max-iter)") << "\n";

        std::cout << "\nMemory has " << mem->messages().size()
                  << " messages stored in SQLite.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "agent failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
