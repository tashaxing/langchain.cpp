// langchain/mcp/mcp_server.h
// Skeleton MCP (Model Context Protocol) server: JSON-RPC 2.0 over HTTP.
//
// Counterpart to McpClient. Wraps a langchain::tool::ToolRegistry and exposes
// it to remote MCP-aware peers (other agents, IDE integrations, etc.) via the
// three handshake methods every spec-compliant client expects:
//
//   initialize     — protocol version + server info
//   tools/list     — enumerate registered tools and their JSON schemas
//   tools/call     — invoke a tool by name with arguments, return text content
//
// Transport is HTTP-only for now (stdio/websocket can be added by following
// the same dispatch shape). Lifecycle hooks fire BeforeTool/AfterTool around
// every tools/call invocation so observers see remote tool traffic the same
// way they see in-process tool traffic from agents.
#pragma once

#include "hook/hook.h"
#include "tool/tool.h"
#include "util/common.h"

#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace langchain
{
namespace mcp
{

struct McpServerConfig
{
    std::string host        = "0.0.0.0";
    int         port        = 8765;
    std::string path        = "/";             // JSON-RPC endpoint
    std::string server_name = "langchain.cpp";
    std::string server_version = "0.1.0";
    std::string protocol_version = "2024-11-05"; // mirrors current MCP spec
    int         read_timeout_sec  = 60;
    int         write_timeout_sec = 60;
};

class McpServer
{
public:
    explicit McpServer(McpServerConfig cfg = {});
    ~McpServer();

    McpServer(const McpServer&)            = delete;
    McpServer& operator=(const McpServer&) = delete;

    // Register a tool. Replaces any prior tool with the same name.
    void register_tool(tool::ToolPtr t);

    // Bulk-register from an existing ToolRegistry (each tool re-exposed by name).
    void register_tools(const tool::ToolRegistry& reg);

    void set_hooks(hook::HookManager* mgr);

    // Blocking listen on the calling thread.
    void run();

    // Background variant; pair with stop().
    void start();
    void stop();
    bool is_running() const;

    const McpServerConfig& config() const
    {
        return cfg_;
    }

private:
    // Build the JSON-RPC reply for a single decoded request. Pulled out so it
    // can be unit-tested without standing up the HTTP server.
    json dispatch(const json& request);

    struct Impl;
    std::unique_ptr<Impl> impl_;

    McpServerConfig                cfg_;
    mutable std::mutex             mu_;
    tool::ToolRegistry             tools_;
    hook::HookManager*             hooks_ = nullptr;
    std::thread                    worker_;
};

} // namespace mcp
} // namespace langchain
