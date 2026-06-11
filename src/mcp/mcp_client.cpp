#include "mcp/mcp_client.h"

#include <httplib.h>

namespace langchain
{
namespace mcp
{

namespace
{

struct ParsedUrl
{
    std::string scheme_host;
    std::string path_prefix;
};

ParsedUrl split(const std::string& url)
{
    ParsedUrl o;
    auto pos = url.find("://");
    if (pos == std::string::npos)
    {
        o.scheme_host = url;
        return o;
    }
    auto path_pos = url.find('/', pos + 3);
    if (path_pos == std::string::npos)
    {
        o.scheme_host = url;
        return o;
    }
    o.scheme_host = url.substr(0, path_pos);
    o.path_prefix = url.substr(path_pos);
    return o;
}

} // namespace

json HttpTransport::request(const std::string& method, const json& params)
{
    auto p = split(base_url_);
    httplib::Client cli(p.scheme_host);
    cli.set_read_timeout(60);

    json envelope = {
        {"jsonrpc", "2.0"},
        {"id", next_id_++},
        {"method", method},
        {"params", params}
    };
    auto target = p.path_prefix + path_;
    auto res = cli.Post(target.c_str(), envelope.dump(), "application/json");
    if (!res)
    {
        throw LCError("McpClient: transport error");
    }
    if (res->status / 100 != 2)
    {
        throw LCError("McpClient: HTTP " + std::to_string(res->status) + ": " + res->body);
    }
    json reply = json::parse(res->body, nullptr, false);
    if (reply.is_discarded())
    {
        throw LCError("McpClient: bad JSON reply");
    }
    if (reply.contains("error"))
    {
        throw LCError("McpClient: rpc error: " + reply["error"].dump());
    }
    return reply.value("result", json::object());
}

std::vector<McpClient::ToolInfo> McpClient::list_tools()
{
    auto result = transport_->request("tools/list", json::object());
    std::vector<ToolInfo> out;
    for (const auto& t : result.value("tools", json::array()))
    {
        ToolInfo info;
        info.name         = t.value("name", std::string());
        info.description  = t.value("description", std::string());
        info.input_schema = t.value("inputSchema", json::object());
        out.push_back(std::move(info));
    }
    return out;
}

std::string McpClient::invoke_tool(const std::string& name, const json& arguments)
{
    json params = {{"name", name}, {"arguments", arguments}};
    auto result = transport_->request("tools/call", params);
    std::string out;
    for (const auto& block : result.value("content", json::array()))
    {
        if (block.value("type", std::string()) == "text")
        {
            out += block.value("text", std::string());
            out += "\n";
        }
    }
    return out;
}

std::vector<tool::ToolPtr> McpClient::as_lc_tools()
{
    std::vector<tool::ToolPtr> out;
    auto self = transport_;
    for (auto& t : list_tools())
    {
        std::string name = t.name;
        out.push_back(std::make_shared<tool::FunctionTool>(
            t.name, t.description, t.input_schema,
            [self, name](const json& args) -> std::string
            {
                json params = {{"name", name}, {"arguments", args}};
                auto result = self->request("tools/call", params);
                std::string s;
                for (const auto& block : result.value("content", json::array()))
                {
                    if (block.value("type", std::string()) == "text")
                    {
                        s += block.value("text", std::string());
                    }
                }
                return s;
            }));
    }
    return out;
}

} // namespace mcp
} // namespace langchain
