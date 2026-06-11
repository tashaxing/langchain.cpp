// examples/16_mcp_client_server.cpp — MCP (Model Context Protocol) server + client.
//
// Demonstrates:
//   1. Starting an McpServer with registered tools.
//   2. Connecting via McpClient over HTTP transport.
//   3. Listing remote tools and invoking them.
//   4. Wrapping remote tools as local langchain::tool::ITool instances.
//
// No external LLM required. Build & run:
//   cmake --build build
//   ./build/16_mcp_client_server
#include "langchain.h"

#include <atomic>
#include <chrono>
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

int main()
{
    using namespace langchain;

    std::signal(SIGINT, on_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, on_signal);
#endif

    // ---- 1. Start MCP server with tools ----
    mcp::McpServerConfig srv_cfg;
    srv_cfg.host = "127.0.0.1";
    srv_cfg.port = 8765;

    mcp::McpServer server(srv_cfg);
    server.register_tool(tool::make_calculator_tool());
    server.register_tool(tool::make_http_get_tool());
    server.start();

    std::cout << "MCP server listening on http://"
              << srv_cfg.host << ":" << srv_cfg.port << "\n";

    // Give the server time to bind.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // ---- 2. Connect as MCP client ----
    auto transport = std::make_shared<mcp::HttpTransport>(
        "http://127.0.0.1:" + std::to_string(srv_cfg.port));
    mcp::McpClient client(transport);

    try
    {
        // ---- 3. List remote tools ----
        std::cout << "\n=== Remote tools ===\n";
        auto tool_infos = client.list_tools();
        for (const auto& t : tool_infos)
        {
            std::cout << "  - " << t.name << ": " << t.description << "\n";
        }

        // ---- 4. Invoke a remote tool directly ----
        std::cout << "\n=== Invoke calculator via MCP ===\n";
        json args = {{"expression", "(1 + 2) * 3 - 4 / 2"}};
        std::string result = client.invoke_tool("calculator", args);
        std::cout << "Result: " << result << "\n";

        // ---- 5. Wrap remote tools as local ITool ----
        std::cout << "\n=== Wrapped as local tools ===\n";
        auto local_tools = client.as_lc_tools();
        for (const auto& t : local_tools)
        {
            std::cout << "  Local tool: " << t->name() << "\n";
        }

        if (!local_tools.empty())
        {
            auto out = local_tools[0]->invoke(args);
            std::cout << "  Invocation result: " << out << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "MCP client error: " << e.what() << "\n";
    }

    std::cout << "\nPress Ctrl-C to stop.\n";
    while (!g_stop.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "Stopping MCP server...\n";
    server.stop();
    return 0;
}
