// langchain/a2a/a2a_client.h
// A2A client: discover remote agents, send tasks, and receive results.
#pragma once

#include "a2a/a2a.h"

#include <functional>
#include <string>
#include <vector>

namespace langchain
{
namespace a2a
{

// Callback invoked per streamed task update. Return false to abort.
using TaskUpdateCallback = std::function<bool(const Task& task)>;

// A2A client for interacting with remote agents.
class A2AClient
{
public:
    explicit A2AClient(std::string base_url,
                       std::string agent_path = "/");
    ~A2AClient();

    A2AClient(const A2AClient&)            = delete;
    A2AClient& operator=(const A2AClient&) = delete;

    // Discover the remote agent's capabilities.
    AgentCard discover() const;

    // Send a task and wait for the final result (synchronous).
    Task send_task(const Task& task) const;

    // Send a task and receive streaming updates via callback.
    // The callback is invoked for each status/artifact update.
    void send_task_stream(const Task& task,
                          const TaskUpdateCallback& on_update) const;

    // Retrieve the current status of a task.
    Task get_task(const std::string& task_id) const;

    // Cancel a running task.
    void cancel_task(const std::string& task_id) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace a2a
} // namespace langchain
