// examples/15_tool_calling_agent.cpp — ToolCallingAgent with OpenAI-style tool_calls.
//
// Unlike ReActAgent (text protocol), ToolCallingAgent uses the LLM's native
// tool-calling capability (OpenAI function calling, Ollama tools, etc.).
// This example demonstrates:
//   1. Basic tool invocation with calculator.
//   2. Streaming agent events (thought, tool_call, observation, answer).
//   3. Multimodal message input (text + image).
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
    tools.add(tool::make_http_get_tool());

    agent::AgentConfig acfg;
    acfg.max_iterations = 6;
    agent::ToolCallingAgent agent(llm, std::move(tools), acfg);

    try
    {
        // ---- Demo 1: Basic tool calling ----
        std::cout << "=== Demo 1: Basic tool calling ===\n";
        auto r1 = agent.invoke("What is (17 * 23) + 9?");
        std::cout << "Answer: " << r1.output << "\n";
        std::cout << "Steps: " << r1.steps.size()
                  << (r1.finished ? " (finished)" : " (max-iter)") << "\n";
        for (const auto& s : r1.steps)
        {
            std::cout << "  tool=" << s.tool_name
                      << " input=" << s.tool_input
                      << " observation=" << s.observation << "\n";
        }

        // ---- Demo 2: Streaming agent events ----
        std::cout << "\n=== Demo 2: Streaming agent events ===\n";
        auto r2 = agent.invoke_stream("What is 2 + 3?",
            [](const agent::AgentStreamEvent& ev) -> bool
        {
            switch (ev.type)
            {
            case agent::AgentStreamEventType::Thought:
                std::cout << "[THOUGHT] " << ev.text;
                break;
            case agent::AgentStreamEventType::ToolCall:
                std::cout << "[TOOL_CALL] " << ev.tool_name
                          << "(" << ev.tool_input << ")\n";
                break;
            case agent::AgentStreamEventType::Observation:
                std::cout << "[OBSERVATION] " << ev.text << "\n";
                break;
            case agent::AgentStreamEventType::Answer:
                std::cout << "[ANSWER] " << ev.text << "\n";
                break;
            case agent::AgentStreamEventType::Error:
                std::cout << "[ERROR] " << ev.text << "\n";
                break;
            case agent::AgentStreamEventType::Done:
                std::cout << "[DONE]\n";
                break;
            default:
                break;
            }
            return true;
        });
        std::cout << "Final answer: " << r2.output << "\n";

        // ---- Demo 3: Multimodal input ----
        std::cout << "\n=== Demo 3: Multimodal input ===\n";
        Message mm = Message::user_with_image(
            "What do you see in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png");
        auto r3 = agent.invoke(mm);
        std::cout << "Answer: " << r3.output << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "agent failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
