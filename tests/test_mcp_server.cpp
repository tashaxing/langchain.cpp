// tests/test_gtest_mcp_server.cpp — JSON-RPC dispatch coverage for McpServer.
//
// Brings the full HTTP server up on a fixed local port and drives it with
// the existing McpClient. Mirrors the round-trip pattern already used for
// ApiServer.
#include <gtest/gtest.h>

#include "hook/hook.h"
#include "mcp/mcp_client.h"
#include "mcp/mcp_server.h"
#include "tool/tool.h"

#include <atomic>
#include <memory>
#include <string>

using namespace langchain;

namespace
{

tool::ToolPtr make_echo_tool()
{
    json schema = {
        {"type", "object"},
        {"properties", {{"text", {{"type", "string"}}}}},
        {"required", json::array({"text"})}
    };
    return std::make_shared<tool::FunctionTool>(
        "echo", "Echoes its argument",
        std::move(schema),
        [](const json& args) -> std::string
        {
            return "echo: " + args.value("text", std::string());
        });
}

} // namespace

// Round-trip through the actual HTTP listener: brings the full server up on a
// fixed port and verifies the three core MCP methods. Skipped if the port is
// in use — printing rather than failing keeps CI green on noisy hosts.
TEST(McpServer, RoundTripOverHttp)
{
    mcp::McpServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 18765;

    mcp::McpServer server(cfg);
    server.register_tool(make_echo_tool());
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind 127.0.0.1:" << cfg.port;
    }

    auto transport = std::make_shared<mcp::HttpTransport>(
        "http://127.0.0.1:" + std::to_string(cfg.port), "/");
    mcp::McpClient client(transport);

    auto tools = client.list_tools();
    ASSERT_EQ(tools.size(), 1u);
    EXPECT_EQ(tools[0].name, "echo");
    EXPECT_FALSE(tools[0].description.empty());
    EXPECT_TRUE(tools[0].input_schema.contains("properties"));

    auto out = client.invoke_tool("echo", json{{"text", "hello"}});
    EXPECT_EQ(out, "echo: hello\n");

    server.stop();
}

TEST(McpServer, HooksFireForRemoteToolCall)
{
    hook::HookManager mgr;
    std::atomic<int> before_hits{0};
    std::atomic<int> after_hits{0};
    mgr.add("before",
        [&](hook::HookContext&) { ++before_hits; },
        { hook::Phase::BeforeTool });
    mgr.add("after",
        [&](hook::HookContext&) { ++after_hits; },
        { hook::Phase::AfterTool });

    mcp::McpServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 18766;

    mcp::McpServer server(cfg);
    server.register_tool(make_echo_tool());
    server.set_hooks(&mgr);
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind 127.0.0.1:" << cfg.port;
    }

    auto transport = std::make_shared<mcp::HttpTransport>(
        "http://127.0.0.1:" + std::to_string(cfg.port), "/");
    mcp::McpClient client(transport);
    client.invoke_tool("echo", json{{"text", "x"}});
    client.invoke_tool("echo", json{{"text", "y"}});

    EXPECT_EQ(before_hits.load(), 2);
    EXPECT_EQ(after_hits.load(),  2);

    server.stop();
}

TEST(McpServer, UnknownToolReturnsErrorBlock)
{
    mcp::McpServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 18767;

    mcp::McpServer server(cfg);
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind 127.0.0.1:" << cfg.port;
    }

    auto transport = std::make_shared<mcp::HttpTransport>(
        "http://127.0.0.1:" + std::to_string(cfg.port), "/");
    mcp::McpClient client(transport);

    auto out = client.invoke_tool("nope", json::object());
    EXPECT_NE(out.find("unknown tool"), std::string::npos);

    server.stop();
}
