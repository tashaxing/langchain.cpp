// langchain/mcp/mcp_client.h
// Skeleton MCP (Model Context Protocol) client: JSON-RPC 2.0 over a transport.
// Only an HTTP transport is provided; add stdio/websocket as needed.
#pragma once

#include "tool/tool.h"
#include "util/common.h"

#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace mcp
{

// Transport abstraction (stdio, HTTP, websocket, etc.).
class ITransport
{
public:
    virtual ~ITransport() = default;
    virtual json request(const std::string& method, const json& params) = 0;
};

using TransportPtr = std::shared_ptr<ITransport>;

// HTTP JSON-RPC transport — POSTs {jsonrpc,id,method,params} and reads back result.
class HttpTransport : public ITransport
{
public:
    HttpTransport(std::string base_url, std::string path = "/")
        : base_url_(std::move(base_url)),
          path_(std::move(path))
    {
    }

    json request(const std::string& method, const json& params) override;

private:
    std::string base_url_;
    std::string path_;
    int next_id_ = 1;
};

// MCP client. After connecting, list_tools() returns the server's tool catalog;
// invoke_tool() calls one. as_lc_tools() wraps each as a langchain::tool::ITool
// so they can be registered with any agent.
class McpClient
{
public:
    explicit McpClient(TransportPtr transport)
        : transport_(std::move(transport))
    {
    }

    struct ToolInfo
    {
        std::string name;
        std::string description;
        json input_schema;
    };

    std::vector<ToolInfo> list_tools();
    std::string invoke_tool(const std::string& name, const json& arguments);

    // Wrap remote tools as local ITool instances.
    std::vector<tool::ToolPtr> as_lc_tools();

private:
    TransportPtr transport_;
};

} // namespace mcp
} // namespace langchain
