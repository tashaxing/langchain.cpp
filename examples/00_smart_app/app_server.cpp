// app_server.cpp -- HTTP server builder and custom route handlers.
#include "app_server.h"
#include "app_config.h"
#include "app_tools.h"
#include "app_hooks.h"
#include "app_rag.h"
#include "app_agent.h"

#include "util/config.h"
#include "util/logging.h"
#include "util/singleton.h"

#include <filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace smart_app
{

namespace
{

// Extract X-Session-Id header or return default.
std::string get_session_id(const langchain::api::Request& req)
{
    auto it = req.headers.find("X-Session-Id");
    if (it != req.headers.end() && !it->second.empty())
    {
        return it->second;
    }
    return get_memory_config().default_session_id;
}

} // namespace

AppServer::AppServer() = default;
AppServer::~AppServer() = default;

void AppServer::build()
{
    using namespace langchain;

    // ---- API Server ----
    auto api_cfg = get_api_config();
    server_ = std::make_unique<api::ApiServer>(api_cfg);
    LOG_INFO("ApiServer configured: host={} port={}", api_cfg.host, api_cfg.port);

    // ---- LLM Backends ----
    auto llm_cfg = get_llm_config();
    std::vector<std::string> models = get_llm_models();
    if (models.empty())
    {
        LOG_WARN("No models configured in [llm] section; falling back to default_model");
        models.push_back(llm_cfg.model);
    }

    std::unordered_map<std::string, llm::LLMPtr> llms;
    for (const auto& model : models)
    {
        auto model_cfg = llm_cfg;
        model_cfg.model = model;
        auto model_llm = std::make_shared<llm::OpenAILLM>(model_cfg);
        llms[model] = model_llm;
        server_->register_model(model, model_llm);
        LOG_INFO("Registered model backend: base_url={} model={}", model_cfg.base_url, model);
    }

    auto default_llm_it = llms.find(llm_cfg.model);
    llm::LLMPtr default_llm = default_llm_it == llms.end() ? llms.begin()->second : default_llm_it->second;

    // ---- RAG ----
    RagConfig rcfg = get_rag_config();
    if (rcfg.enabled)
    {
        vectorstore_ = build_vectorstore();
        if (vectorstore_)
        {
            if (fs::exists(rcfg.knowledge_dir))
            {
                std::size_t n = ingest_documents(rcfg.knowledge_dir);
                LOG_INFO("RAG auto-ingested {} chunks from {}", n, rcfg.knowledge_dir);
            }
            else
            {
                LOG_INFO("RAG knowledge dir not found (will create on ingest): {}", rcfg.knowledge_dir);
            }
        }
    }

    // ---- Memory ----
    MemoryConfig mcfg = get_memory_config();
    memory::MemoryPtr mem = nullptr;
    if (mcfg.enabled)
    {
        LOG_INFO("Memory enabled: backend={} db={}", mcfg.backend, mcfg.db_path);
    }

    // ---- Agents ----
    langchain::agent::AgentConfig acfg = get_agent_config();
    SkillConfig scfg = get_skill_config();
    for (const auto& model : models)
    {
        auto llm_for_model = llms.at(model);
        auto tools = build_tool_registry();
        skill::SkillRegistry skills;
        if (scfg.enabled && fs::exists(scfg.skills_dir))
        {
            try
            {
                skill::SkillLoader::load_directory(scfg.skills_dir, skills, llm_for_model, nullptr);
                LOG_INFO("Loaded {} skill(s) for model {}", skills.size(), model);
            }
            catch (const std::exception& e)
            {
                LOG_WARN("Skill loading failed for model {}: {}", model, e.what());
            }
        }

        auto agent = build_agent(llm_for_model, std::move(tools), std::move(skills), mem);
        server_->register_agent(model, agent);
        LOG_INFO("Agent registered for model: {}", model);
    }

    // ---- Hooks ----
    register_hooks();
    server_->set_hooks(&hook::HookManager::global());

    // ---- Per-request memory ----
    // Hand the API server a resolver that produces a per-session SQLite-backed
    // memory for each /v1/chat/completions request. The server then loads
    // history before invoking the agent and writes the new turn back, so the
    // Agent itself stays stateless and a single shared Agent can serve many
    // independent sessions concurrently.
    if (mcfg.enabled)
    {
        std::string db_path = mcfg.db_path;
        std::string default_sid = mcfg.default_session_id;
        server_->set_memory_resolver(
            [db_path, default_sid](const api::Request& req) -> memory::MemoryPtr
        {
            std::string sid = default_sid;
            auto it = req.headers.find("X-Session-Id");
            if (it != req.headers.end() && !it->second.empty())
            {
                sid = it->second;
            }
            return std::make_shared<memory::LongTermMemory>(
                memory::LongTermMemory::sqlite(db_path, sid));
        });
        LOG_INFO("Memory resolver installed (per-session SQLite at {})", mcfg.db_path);
    }

    // ---- Custom Routes ----
    setup_custom_routes();
}

void AppServer::setup_custom_routes()
{
    using namespace langchain;

    // GET /v1/memory -- list messages for a session.
    server_->add_route("GET", "/v1/memory",
        [this](const api::Request& req, api::Response& res)
    {
        std::string sid = get_session_id(req);
        MemoryConfig mcfg = get_memory_config();

        json arr = json::array();
        if (mcfg.enabled)
        {
            try
            {
                memory::LongTermMemory mem(
                    memory::LongTermMemory::sqlite(mcfg.db_path, sid));
                for (const auto& m : mem.messages())
                {
                    json item;
                    item["role"] = to_string(m.role);

                    // OpenAI-style multimodal output: when content_parts is
                    // populated, return content as an array of {type,...} parts
                    // so the client can render images alongside text. Falls
                    // back to a plain string for text-only messages.
                    if (!m.content_parts.empty())
                    {
                        json parts = json::array();
                        for (const auto& p : m.content_parts)
                        {
                            if (p.type == "text")
                            {
                                parts.push_back({{"type", "text"}, {"text", p.text}});
                            }
                            else if (p.type == "image_url")
                            {
                                parts.push_back({
                                    {"type", "image_url"},
                                    {"image_url", {{"url", p.url}}}});
                            }
                            else if (p.type == "image_base64")
                            {
                                std::string data_url =
                                    "data:" + (p.mime_type.empty() ? "image/png" : p.mime_type)
                                    + ";base64," + p.base64_data;
                                parts.push_back({
                                    {"type", "image_url"},
                                    {"image_url", {{"url", std::move(data_url)}}}});
                            }
                            else
                            {
                                // Unknown part type — fall back to text.
                                parts.push_back({{"type", "text"}, {"text", p.text}});
                            }
                        }
                        item["content"] = std::move(parts);
                    }
                    else
                    {
                        item["content"] = m.content;
                    }

                    if (!m.name.empty()) item["name"] = m.name;
                    arr.push_back(std::move(item));
                }
            }
            catch (const std::exception& e)
            {
                res.status = 500;
                res.set_json({{"error", e.what()}});
                return;
            }
        }
        json out;
        out["session_id"] = sid;
        out["messages"] = arr;
        res.set_json(out);
    });

    // POST /v1/memory/clear -- clear session memory.
    server_->add_route("POST", "/v1/memory/clear",
        [this](const api::Request& req, api::Response& res)
    {
        std::string sid = get_session_id(req);
        MemoryConfig mcfg = get_memory_config();

        if (mcfg.enabled)
        {
            try
            {
                memory::LongTermMemory mem(
                    memory::LongTermMemory::sqlite(mcfg.db_path, sid));
                mem.clear();
            }
            catch (const std::exception& e)
            {
                res.status = 500;
                res.set_json({{"error", e.what()}});
                return;
            }
        }
        res.set_json({{"session_id", sid}, {"status", "cleared"}});
    });

    // GET /v1/config -- view current config sections.
    server_->add_route("GET", "/v1/config",
        [this](const api::Request&, api::Response& res)
    {
        auto& cfg = langchain::util::Singleton<langchain::util::Config>::instance();
        langchain::json sections = langchain::json::object();
        for (const auto& name : cfg.section_names())
        {
            langchain::json kv = langchain::json::object();
            try
            {
                const auto& sec = cfg.section(name);
                for (const auto& item : sec.snapshot())
                {
                    kv[item.first] = item.second;
                }
            }
            catch (const std::exception&)
            {
            }
            sections[name] = std::move(kv);
        }
        res.set_json({{"sections", sections}});
    });

    // POST /v1/config/reload -- trigger hot-reload.
    server_->add_route("POST", "/v1/config/reload",
        [this](const api::Request&, api::Response& res)
    {
        bool ok = reload_app_config();
        res.set_json({{"reloaded", ok}});
    });

    // POST /v1/rag/ingest -- ingest documents into vector store.
    server_->add_route("POST", "/v1/rag/ingest",
        [this](const api::Request& req, api::Response& res)
    {
        if (!vectorstore_)
        {
            res.status = 503;
            res.set_json({{"error", "RAG not enabled or vector store unavailable"}});
            return;
        }

        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded())
        {
            res.status = 400;
            res.set_json({{"error", "invalid JSON"}});
            return;
        }

        std::string dir = body.value("dir", get_rag_config().knowledge_dir);
        std::size_t n = ingest_documents(dir);
        res.set_json({{"ingested_chunks", n}, {"directory", dir}});
    });

    LOG_INFO("Custom routes registered: /v1/memory, /v1/memory/clear, /v1/config, /v1/config/reload, /v1/rag/ingest");
}

void AppServer::start()
{
    if (server_)
    {
        server_->start();
    }
}

void AppServer::stop()
{
    if (server_)
    {
        server_->stop();
    }
}

bool AppServer::is_running() const
{
    return server_ && server_->is_running();
}

} // namespace smart_app
