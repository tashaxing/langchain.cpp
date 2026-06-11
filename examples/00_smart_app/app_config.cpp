// app_config.cpp -- Configuration loading and accessors.
#include "app_config.h"
#include "app_paths.h"

#include "util/config.h"
#include "util/logging.h"
#include "util/singleton.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace smart_app
{

namespace
{

langchain::util::Config& cfg()
{
    return langchain::util::Singleton<langchain::util::Config>::instance();
}

// Ensure a section exists with at least an <enabled> key.
void ensure_section(const std::string& name)
{
    if (!cfg().has_section(name))
    {
        cfg().section(name).set("enabled", "true");
    }
}

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Resolve a config path value: if relative, prepend app base dir.
std::string resolve_path(const std::string& path)
{
    return resolve_app_path(path);
}

} // namespace

void init_app_base_dir()
{
    // Trigger path resolution to validate the base dir is detectable.
    std::string base = get_app_base_dir();
    if (!base.empty())
    {
        std::cout << "[smart_app] App base dir: " << base << "\n";
    }
}

bool load_app_config(const std::string& path)
{
    if (!cfg().load(path))
    {
        LOG_ERROR("[smart_app] Failed to load config: {}", path);
        return false;
    }

    // Ensure all expected sections exist so getters don't throw.
    ensure_section("app");
    ensure_section("llm");
    ensure_section("api_server");
    ensure_section("embedding");
    ensure_section("vectorstore");
    ensure_section("memory");
    ensure_section("rag");
    ensure_section("agent");
    ensure_section("skill");
    ensure_section("tool");

    return true;
}

bool reload_app_config()
{
    return cfg().reload();
}

langchain::llm::OpenAILLMConfig get_llm_config()
{
    const auto& sec = cfg().section("llm");
    langchain::llm::OpenAILLMConfig c;
    c.base_url = sec.get("base_url", c.base_url);
    c.api_key  = sec.get("api_key", c.api_key);
    c.model    = sec.get("default_model", c.model);
    c.connect_timeout_sec = sec.get<int>("connect_timeout_sec", c.connect_timeout_sec);
    c.read_timeout_sec    = sec.get<int>("read_timeout_sec", c.read_timeout_sec);
    return c;
}

std::vector<std::string> get_llm_models()
{
    const auto& sec = cfg().section("llm");
    std::string raw = sec.get("models", std::string());
    if (raw.empty())
    {
        // Fallback: single model from default_model.
        std::string def = sec.get("default_model", std::string());
        if (!def.empty())
        {
            return {def};
        }
        return {};
    }

    std::vector<std::string> out;
    std::size_t pos = 0;
    while (pos < raw.size())
    {
        std::size_t comma = raw.find(',', pos);
        std::string token = raw.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        // Trim whitespace.
        std::size_t start = token.find_first_not_of(" \t\r\n");
        std::size_t end   = token.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos)
        {
            out.push_back(token.substr(start, end - start + 1));
        }
        pos = (comma == std::string::npos) ? raw.size() : comma + 1;
    }
    return out;
}

langchain::api::ApiConfig get_api_config()
{
    const auto& sec = cfg().section("api_server");
    langchain::api::ApiConfig c;
    c.host = sec.get("host", c.host);
    c.port = sec.get<int>("port", c.port);
    c.api_key = sec.get("api_key", c.api_key);
    c.read_timeout_sec  = sec.get<int>("read_timeout_sec", c.read_timeout_sec);
    c.write_timeout_sec = sec.get<int>("write_timeout_sec", c.write_timeout_sec);
    return c;
}

langchain::embedding::HttpEmbeddingConfig get_embedding_config()
{
    const auto& sec = cfg().section("embedding");
    langchain::embedding::HttpEmbeddingConfig c;
    c.base_url = sec.get("base_url", c.base_url);
    c.api_key  = sec.get("api_key", c.api_key);
    c.model    = sec.get("model", c.model);
    c.dimension = sec.get<int>("dimensions", c.dimension);
    c.connect_timeout_sec = sec.get<int>("connect_timeout_sec", c.connect_timeout_sec);
    c.read_timeout_sec    = sec.get<int>("read_timeout_sec", c.read_timeout_sec);
    return c;
}

langchain::agent::AgentConfig get_agent_config()
{
    const auto& sec = cfg().section("agent");
    langchain::agent::AgentConfig c;
    c.max_iterations = sec.get<int>("max_iterations", c.max_iterations);
    c.system_prompt  = sec.get("system_prompt", c.system_prompt);
    if (sec.has("temperature"))
        c.temperature = sec.get<float>("temperature", 0.7f);
    if (sec.has("max_tokens"))
        c.max_tokens = sec.get<int>("max_tokens", 2048);
    if (sec.has("top_p"))
        c.top_p = sec.get<float>("top_p", 0.95f);
    if (sec.has("top_k"))
        c.top_k = sec.get<int>("top_k", 50);
    return c;
}

MemoryConfig get_memory_config()
{
    const auto& sec = cfg().section("memory");
    MemoryConfig c;
    c.enabled = to_lower(sec.get("enabled", "true")) == "true";
    c.backend = sec.get("backend", c.backend);
    c.db_path = resolve_path(sec.get("db_path", c.db_path));
    c.default_session_id = sec.get("default_session_id", c.default_session_id);
    c.max_history_turns = sec.get<int>("max_history_turns", c.max_history_turns);
    return c;
}

RagConfig get_rag_config()
{
    const auto& sec = cfg().section("rag");
    RagConfig c;
    c.enabled = to_lower(sec.get("enabled", "true")) == "true";
    c.knowledge_dir = resolve_path(sec.get("knowledge_dir", c.knowledge_dir));
    c.retriever_top_k = sec.get<int>("retriever_top_k", c.retriever_top_k);
    c.chunk_size = sec.get<int>("chunk_size", c.chunk_size);
    c.chunk_overlap = sec.get<int>("chunk_overlap", c.chunk_overlap);
    return c;
}

LogConfig get_log_config()
{
    const auto& sec = cfg().section("app");
    LogConfig c;
    c.level   = sec.get("log_level", c.level);
    c.dir     = resolve_path(sec.get("log_dir", c.dir));
    c.filename = sec.get("log_filename", c.filename);
    c.max_size = static_cast<std::size_t>(sec.get<int>("log_max_size_mb", 500)) * 1024 * 1024;
    c.max_files = static_cast<std::size_t>(sec.get<int>("log_max_files", 10));
    c.console = to_lower(sec.get("log_console", "true")) == "true";
    return c;
}

ToolConfig get_tool_config()
{
    const auto& sec = cfg().section("tool");
    ToolConfig c;
    c.calculator_enabled = to_lower(sec.get("calculator_enabled", "true")) == "true";
    c.http_get_enabled   = to_lower(sec.get("http_get_enabled", "true")) == "true";
    c.datetime_enabled   = to_lower(sec.get("datetime_enabled", "true")) == "true";
    c.echo_enabled       = to_lower(sec.get("echo_enabled", "true")) == "true";
    return c;
}

SkillConfig get_skill_config()
{
    const auto& sec = cfg().section("skill");
    SkillConfig c;
    c.enabled = to_lower(sec.get("enabled", "true")) == "true";
    c.skills_dir = resolve_path(sec.get("skills_dir", c.skills_dir));
    return c;
}

} // namespace smart_app
