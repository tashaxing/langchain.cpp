// examples/06_api_server.cpp — start an OpenAI-compatible REST server backed
// by an OpenAILLM (or any ILLM), with a logging hook that prints every
// lifecycle event. Conversation context is persisted to SQLite so that
// multi-turn chats remember previous exchanges.
//
// Hit it with:
//   curl http://localhost:8080/v1/models
//   curl http://localhost:8080/v1/chat/completions \
//     -H 'Content-Type: application/json' \
//     -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}'
//
// Multimodal pass-through:
//   curl http://localhost:8080/v1/chat/completions \
//     -H 'Content-Type: application/json' \
//     -d '{"model":"local","messages":[{"role":"user","content":[
//           {"type":"text","text":"What is in this image?"},
//           {"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}
//         ]}]}'
#include "langchain.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
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

    // ---- Hook: log every lifecycle phase to stderr. ----
    auto& hooks = hook::HookManager::global();
    hooks.add("stderr-logger", [](hook::HookContext& ctx)
    {
        std::cerr << "[hook] " << hook::to_string(ctx.phase)
                  << " component=" << ctx.component
                  << " call_id=" << ctx.call_id;
        if (ctx.elapsed.count() > 0)
        {
            std::cerr << " elapsed_us=" << ctx.elapsed.count();
        }
        if (ctx.agent_input)
        {
            std::cerr << " agent_input=\"" << *ctx.agent_input << "\"";
        }
        if (ctx.tool_name)
        {
            std::cerr << " tool=" << *ctx.tool_name;
        }
        std::cerr << "\n";
    });

    // ---- Backend LLM. Defaults work against a local OpenAI-compat endpoint. ----
    llm::OpenAILLMConfig cfg;
    cfg.base_url = std::getenv("LC_BASE_URL") ? std::getenv("LC_BASE_URL")
                                               : "http://localhost:11434";
    cfg.model    = std::getenv("LC_MODEL") ? std::getenv("LC_MODEL") : "qwen2.5:0.5b";
    if (const char* k = std::getenv("LC_API_KEY"))
    {
        cfg.api_key = k;
    }

    auto backend = std::make_shared<llm::OpenAILLM>(cfg);

    api::ApiConfig acfg;
    acfg.host = "0.0.0.0";
    acfg.port = 8080;
    if (const char* p = std::getenv("LC_PORT"))
    {
        acfg.port = std::atoi(p);
    }

    api::ApiServer server(acfg);
    server.register_model("local", backend);
    server.register_model(cfg.model, backend);
    server.set_hooks(&hooks);

    // ---- Persistent SQLite memory per session ----
    // The OpenAI /v1/chat/completions route does not automatically manage
    // memory because each request is stateless. Custom routes can use any
    // memory implementation (BufferMemory, LongTermMemory, etc.) to build
    // stateful agents or assistants.
    std::filesystem::create_directories("build");
    auto mem = std::make_shared<memory::LongTermMemory>(
        memory::LongTermMemory::sqlite("build/api_server_memory.db", "default-session"));
    std::cout << "Memory: " << mem->messages().size()
              << " messages loaded from SQLite.\n";

    // ---- Custom route: inspect persisted memory ----
    server.add_route("GET", "/v1/memory", [mem](const api::Request&, api::Response& res)
    {
        json j = json::array();
        for (const auto& m : mem->messages())
        {
            json item;
            item["role"] = to_string(m.role);
            item["content"] = m.content;
            if (!m.name.empty()) item["name"] = m.name;
            j.push_back(std::move(item));
        }
        res.set_json(j);
    });

    std::signal(SIGINT, on_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, on_signal);
#endif

    server.start();
    std::cout << "langchain.cpp api server listening on "
              << acfg.host << ":" << acfg.port << "\n"
              << "  GET  /v1/models\n"
              << "  POST /v1/chat/completions  (set \"stream\":true for SSE)\n"
              << "  GET  /v1/memory            (list stored messages)\n"
              << "Ctrl-C to quit.\n";

    while (!g_stop.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "shutting down...\n";
    server.stop();
    return 0;
}
