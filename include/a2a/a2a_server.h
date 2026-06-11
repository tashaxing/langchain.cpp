// langchain/a2a/a2a_server.h
// A2A server: expose a local agent so other agents can discover it and
// send tasks.  Built on top of api::ApiServer.
#pragma once

#include "a2a/a2a.h"
#include "api/server.h"

#include <functional>
#include <memory>
#include <string>

namespace langchain
{
namespace a2a
{

// Handler called when a task arrives.  Must return the final Task state.
using TaskHandler = std::function<Task(const Task& incoming)>;

// Handler called when a task arrives with streaming support.
// The handler should write intermediate updates and return the final Task.
using StreamTaskHandler = std::function<Task(const Task& incoming,
                                              std::function<void(const Task&)> update)>;

// A2A server exposing a local agent via the A2A protocol.
// Mounts the standard A2A endpoints onto an existing ApiServer.
class A2AServer
{
public:
    A2AServer(const AgentCard& card,
              const TaskHandler& handler);

    A2AServer(const AgentCard& card,
              const StreamTaskHandler& handler);

    ~A2AServer();

    A2AServer(const A2AServer&)            = delete;
    A2AServer& operator=(const A2AServer&) = delete;

    // Mount A2A routes onto an existing ApiServer.
    void mount(api::ApiServer& server);

    // Convenience: start a dedicated server on the given port.
    void start(int port);
    void stop();

    // True between start() and stop().
    bool is_running() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace a2a
} // namespace langchain
