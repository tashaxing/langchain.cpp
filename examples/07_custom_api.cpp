// examples/07_custom_api.cpp -- general-purpose REST API with custom routes,
// path parameters, query parameters, and hook observability.
//
// No LLM required. Start it and hit:
//   curl http://localhost:8080/healthz
//   curl http://localhost:8080/api/users
//   curl http://localhost:8080/api/users/42
//   curl http://localhost:8080/api/users/42/posts/7?active=1
//
// Or with a POST:
//   curl -X POST http://localhost:8080/api/echo \
//     -H 'Content-Type: application/json' \
//     -d '{"msg":"hello"}'
#include "langchain.h"

#include <atomic>
#include <csignal>
#include <chrono>
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

    // ---- Hook: log every API lifecycle event. ----
    auto& hooks = hook::HookManager::global();
    hooks.add("api-logger", [](hook::HookContext& ctx)
    {
        std::cerr << "[hook] " << hook::to_string(ctx.phase)
                  << " component=" << ctx.component
                  << " call_id=" << ctx.call_id;
        if (ctx.elapsed.count() > 0)
        {
            std::cerr << " elapsed_us=" << ctx.elapsed.count();
        }
        if (ctx.metadata.contains("route"))
        {
            std::cerr << " route=" << ctx.metadata["route"].get<std::string>();
        }
        if (ctx.metadata.contains("status"))
        {
            std::cerr << " status=" << ctx.metadata["status"].get<int>();
        }
        std::cerr << "\n";
    });

    api::ApiConfig cfg;
    cfg.host = "0.0.0.0";
    cfg.port = 8080;
    if (const char* p = std::getenv("LC_PORT"))
    {
        cfg.port = std::atoi(p);
    }

    api::ApiServer server(cfg);
    server.set_hooks(&hooks);

    // ---- Custom route: list users ----
    server.add_route("GET", "/api/users",
        [](const api::Request& /*req*/, api::Response& res)
    {
        json users = json::array({
            {{"id", "1"}, {"name", "Alice"}},
            {{"id", "2"}, {"name", "Bob"}}
        });
        res.set_json({{"users", users}});
    });

    // ---- Custom route: get user by id ----
    server.add_route("GET", "/api/users/:id",
        [](const api::Request& req, api::Response& res)
    {
        auto it = req.path_params.find("id");
        if (it == req.path_params.end())
        {
            res.status = 400;
            res.set_json({{"error", "missing id"}});
            return;
        }
        res.set_json({
            {"id", it->second},
            {"name", "User " + it->second}
        });
    });

    // ---- Custom route: nested path params + query params ----
    server.add_route("GET", "/api/users/:uid/posts/:pid",
        [](const api::Request& req, api::Response& res)
    {
        json out;
        out["user_id"]  = req.path_params.at("uid");
        out["post_id"]  = req.path_params.at("pid");
        out["queries"]  = req.query_params;
        res.set_json(out);
    });

    // ---- Custom route: echo POST body ----
    server.add_route("POST", "/api/echo",
        [](const api::Request& req, api::Response& res)
    {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded())
        {
            res.status = 400;
            res.set_json({{"error", "invalid JSON"}});
            return;
        }
        res.set_json({{"echo", body}});
    });

    // ---- Custom route: SSE streaming ----
    // Test with: curl http://localhost:8080/api/stream
    server.add_route("GET", "/api/stream",
        [](const api::Request& /*req*/, api::Response& res)
    {
        res.enable_streaming([](api::StreamSink& sink)
        {
            for (int i = 1; i <= 5; ++i)
            {
                json chunk = {
                    {"index", i},
                    {"message", "This is chunk " + std::to_string(i)}
                };
                if (!sink.write(chunk.dump()))
                {
                    break;  // client disconnected
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            sink.done();
        });
    });

    std::signal(SIGINT, on_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, on_signal);
#endif

    server.start();
    std::cout << "Custom API server listening on "
              << cfg.host << ":" << cfg.port << "\n"
              << "  GET  /healthz\n"
              << "  GET  /api/users\n"
              << "  GET  /api/users/:id\n"
              << "  GET  /api/users/:uid/posts/:pid\n"
              << "  POST /api/echo\n"
              << "  GET  /api/stream        (SSE)\n"
              << "Ctrl-C to quit.\n";

    while (!g_stop.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "shutting down...\n";
    server.stop();
    return 0;
}
