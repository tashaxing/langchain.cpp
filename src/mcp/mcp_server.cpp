// src/mcp/mcp_server.cpp — JSON-RPC 2.0 over HTTP, MCP method set.
#include "mcp/mcp_server.h"

#include "util/logging.h"

#include <httplib.h>

#include <chrono>
#include <condition_variable>
#include <utility>

namespace langchain
{
namespace mcp
{

namespace
{

// JSON-RPC error codes per spec.
constexpr int kParseError     = -32700;
constexpr int kInvalidRequest = -32600;
constexpr int kMethodNotFound = -32601;
constexpr int kInternalError  = -32603;

json make_error(const json& id, int code, const std::string& msg)
{
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {{"code", code}, {"message", msg}}}
    };
}

json make_result(const json& id, json result)
{
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", std::move(result)}
    };
}

// MCP tools/call returns a `content` array of typed blocks. We always emit a
// single text block — mirrors the client's parser in mcp_client.cpp.
json text_content(const std::string& text, bool is_error = false)
{
    return {
        {"content", json::array({
            {{"type", "text"}, {"text", text}}
        })},
        {"isError", is_error}
    };
}

} // namespace

// ---------------- McpServer::Impl ----------------

struct McpServer::Impl
{
    httplib::Server         svr;
    std::mutex              run_mu;
    std::condition_variable run_cv;
    bool                    running = false;
};

McpServer::McpServer(McpServerConfig cfg)
    : impl_(std::make_unique<Impl>()), cfg_(std::move(cfg))
{
}

McpServer::~McpServer()
{
    stop();
}

void McpServer::register_tool(tool::ToolPtr t)
{
    if (!t)
    {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    tools_.add(std::move(t));
}

void McpServer::register_tools(const tool::ToolRegistry& reg)
{
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& n : reg.names())
    {
        if (auto t = reg.get(n))
        {
            tools_.add(t);
        }
    }
}

void McpServer::set_hooks(hook::HookManager* mgr)
{
    std::lock_guard<std::mutex> lk(mu_);
    hooks_ = mgr;
}

bool McpServer::is_running() const
{
    std::lock_guard<std::mutex> lk(impl_->run_mu);
    return impl_->running;
}

void McpServer::stop()
{
    if (!impl_)
    {
        return;
    }
    // Join the worker before returning so a second stop() (e.g. from the
    // destructor) sees httplib's is_running_ already cleared — otherwise the
    // second svr.stop() can race with listen_internal()'s scope_exit and
    // assert on svr_sock_ == INVALID_SOCKET.
    impl_->svr.stop();
    if (worker_.joinable())
    {
        worker_.join();
    }
    std::lock_guard<std::mutex> lk(impl_->run_mu);
    impl_->running = false;
    impl_->run_cv.notify_all();
}

void McpServer::start()
{
    if (worker_.joinable())
    {
        bool running = false;
        {
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            running = impl_->running;
        }
        if (running)
        {
            throw LCError("McpServer: already started");
        }
        worker_.join();
    }

    worker_ = std::thread([this]
    {
        try
        {
            run();
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("McpServer worker stopped after exception: {}", e.what());
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            impl_->running = false;
            impl_->run_cv.notify_all();
        }
        catch (...)
        {
            LOG_ERROR("McpServer worker stopped after unknown exception");
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            impl_->running = false;
            impl_->run_cv.notify_all();
        }
    });

    // cpp-httplib's wait_until_ready spin-waits until the listen socket is
    // accepting. Using it (instead of our own cv) avoids a race where stop()
    // beats listen() to the socket and asserts inside httplib.
    impl_->svr.wait_until_ready();

    std::lock_guard<std::mutex> lk(impl_->run_mu);
    impl_->running = true;
    impl_->run_cv.notify_all();
}

json McpServer::dispatch(const json& req)
{
    json id = req.contains("id") ? req["id"] : json(nullptr);

    if (!req.is_object() || req.value("jsonrpc", std::string()) != "2.0"
        || !req.contains("method") || !req["method"].is_string())
    {
        return make_error(id, kInvalidRequest, "invalid JSON-RPC 2.0 request");
    }

    const std::string method = req["method"].get<std::string>();
    const json params = req.value("params", json::object());

    if (method == "initialize")
    {
        json result = {
            {"protocolVersion", cfg_.protocol_version},
            {"capabilities", {
                {"tools", {{"listChanged", false}}}
            }},
            {"serverInfo", {
                {"name",    cfg_.server_name},
                {"version", cfg_.server_version}
            }}
        };
        return make_result(id, std::move(result));
    }

    if (method == "tools/list")
    {
        json arr = json::array();
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& n : tools_.names())
            {
                auto t = tools_.get(n);
                if (!t)
                {
                    continue;
                }
                arr.push_back({
                    {"name",        t->name()},
                    {"description", t->description()},
                    {"inputSchema", t->parameters_schema()}
                });
            }
        }
        return make_result(id, json{{"tools", std::move(arr)}});
    }

    if (method == "tools/call")
    {
        const std::string name = params.value("name", std::string());
        const json args        = params.value("arguments", json::object());

        tool::ToolPtr t;
        {
            std::lock_guard<std::mutex> lk(mu_);
            t = tools_.get(name);
        }
        if (!t)
        {
            return make_result(id, text_content("error: unknown tool '" + name + "'", true));
        }

        auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();

        hook::HookContext before;
        before.phase          = hook::Phase::BeforeTool;
        before.component      = t->name();
        before.call_id        = mgr->new_call_id();
        before.tool_name      = &name;
        before.tool_arguments = &args;
        hook::ScopedSpan span(mgr, before, hook::Phase::AfterTool);

        std::string observation;
        bool is_err = false;
        try
        {
            observation = t->invoke(args);
        }
        catch (const std::exception& e)
        {
            observation = std::string("error: ") + e.what();
            is_err = true;
        }
        span.after().tool_name        = &name;
        span.after().tool_observation = &observation;

        return make_result(id, text_content(observation, is_err));
    }

    // ping is part of the spec's keepalive surface; cheap to support.
    if (method == "ping")
    {
        return make_result(id, json::object());
    }

    return make_error(id, kMethodNotFound, "method not found: " + method);
}

void McpServer::run()
{
    auto& svr = impl_->svr;

    svr.Post(cfg_.path.c_str(), [this](const httplib::Request& req, httplib::Response& res)
    {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded())
        {
            res.status = 400;
            res.set_content(
                make_error(nullptr, kParseError, "invalid JSON").dump(),
                "application/json");
            return;
        }

        try
        {
            // Batched requests: an array of envelopes returns an array of replies.
            if (body.is_array())
            {
                json out = json::array();
                for (const auto& one : body)
                {
                    out.push_back(dispatch(one));
                }
                res.set_content(out.dump(), "application/json");
                return;
            }
            res.set_content(dispatch(body).dump(), "application/json");
        }
        catch (const std::exception& e)
        {
            res.status = 500;
            res.set_content(
                make_error(body.value("id", json(nullptr)),
                           kInternalError, e.what()).dump(),
                "application/json");
        }
    });

    svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res)
    {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.set_read_timeout(cfg_.read_timeout_sec, 0);
    svr.set_write_timeout(cfg_.write_timeout_sec, 0);

    svr.listen(cfg_.host.c_str(), cfg_.port);

    {
        std::lock_guard<std::mutex> lk(impl_->run_mu);
        impl_->running = false;
    }
}

} // namespace mcp
} // namespace langchain
