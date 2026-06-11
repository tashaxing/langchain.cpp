// examples/10_a2a_agent.cpp -- A2A server + client demo.
// Starts a local A2A-capable agent, then connects to it from another
// A2A client and sends a task.
//
// Build & run:
//   cmake --build build
//   ./build/10_a2a_agent                        # Linux/macOS
//   cmake --build build_x64 --config Debug
//   ./build_x64/Debug/10_a2a_agent.exe          # Windows
//
// Test with curl:
//   curl http://localhost:9099/.well-known/agent.json
//   curl -X POST http://localhost:9099/tasks/send \
//     -H 'Content-Type: application/json' \
//     -d '{"id":"task-1","status":{"state":"pending"},"artifacts":[],"history":[]}'
#include "langchain.h"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace
{
    std::atomic<bool> g_stop{false};
    void on_signal(int)
    {
        g_stop.store(true);
    }
} // namespace

using namespace langchain;

// Build a simple AgentCard for our demo agent.
a2a::AgentCard make_demo_card()
{
    a2a::AgentCard card;
    card.name        = "demo-agent";
    card.description = "A demo A2A agent that echoes back user messages";
    card.url         = "http://localhost:9099";
    card.version     = "1.0.0";
    card.capabilities.streaming = true;

    a2a::AgentSkill skill;
    skill.id          = "echo";
    skill.name        = "Echo";
    skill.description = "Echoes the input message back to the caller";
    skill.input_modes  = {"text"};
    skill.output_modes = {"text"};
    card.skills.push_back(std::move(skill));

    return card;
}

// Build a task that contains a single user message.
a2a::Task make_echo_task(const std::string& text)
{
    a2a::Task task;
    task.id         = "task-" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    task.status.state = a2a::TaskState::Pending;
    task.history.push_back(a2a::Message::text("user", text));
    return task;
}

int main()
{
    std::signal(SIGINT, on_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, on_signal);
#endif

    // ---- Start an A2A server with a streaming task handler ----
    a2a::AgentCard card = make_demo_card();

    a2a::A2AServer server(card,
        [](const a2a::Task& incoming,
           std::function<void(const a2a::Task&)> update)
        {
            a2a::Task result = incoming;
            result.status.state = a2a::TaskState::Working;

            // Intermediate update
            a2a::Task intermediate = result;
            intermediate.status.state = a2a::TaskState::Working;
            update(intermediate);

            // Final result: echo the last user message back
            std::string last_msg;
            for (const auto& m : result.history)
            {
                if (!m.parts.empty() && m.parts[0])
                {
                    if (m.parts[0]->type() == a2a::PartType::Text)
                    {
                        last_msg = static_cast<a2a::TextPart*>(m.parts[0].get())->text;
                    }
                }
            }

            a2a::Artifact artifact;
            artifact.name = "echo-output";
            artifact.parts.push_back(std::make_unique<a2a::TextPart>(
                "Echo: " + last_msg));
            result.artifacts.push_back(std::move(artifact));
            result.status.state = a2a::TaskState::Completed;
            return result;
        });

    std::cout << "Starting A2A server on port 9099...\n";
    server.start(9099);

    // Give the server a moment to bind.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // ---- Connect as an A2A client and discover the agent ----
    std::cout << "Connecting as A2A client...\n";
    a2a::A2AClient client("http://localhost:9099");

    try
    {
        a2a::AgentCard remote_card = client.discover();
        std::cout << "Discovered agent: " << remote_card.name
                  << " - " << remote_card.description << "\n";

        // ---- Send a task synchronously ----
        a2a::Task task = make_echo_task("Hello from A2A!");
        std::cout << "Sending task: " << task.id << "\n";
        a2a::Task result = client.send_task(task);
        std::cout << "Task completed with state: "
                  << a2a::to_string(result.status.state) << "\n";
        for (const auto& a : result.artifacts)
        {
            std::cout << "  Artifact: " << a.name << "\n";
            for (const auto& p : a.parts)
            {
                if (p && p->type() == a2a::PartType::Text)
                {
                    std::cout << "    "
                              << static_cast<a2a::TextPart*>(p.get())->text
                              << "\n";
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "A2A client error: " << e.what() << "\n";
    }

    std::cout << "Press Ctrl-C to stop.\n";
    while (!g_stop.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "Stopping A2A server...\n";
    server.stop();
    return 0;
}
