// app_config.h -- Configuration helpers for 00_smart_app.
#pragma once

#include "langchain.h"

#include <string>

namespace smart_app
{

// Initialize the application base directory from the executable path.
// Call once at startup before any other path resolution.
void init_app_base_dir();

// Load the XML configuration and set defaults for missing keys.
bool load_app_config(const std::string& path);

// Re-read config from disk if changed.
bool reload_app_config();

// LLM configuration from the <llm> section.
langchain::llm::OpenAILLMConfig get_llm_config();

// List of model IDs from the <llm> section (comma-separated).
std::vector<std::string> get_llm_models();

// API server configuration from the <api_server> section.
langchain::api::ApiConfig get_api_config();

// Embedding configuration from the <embedding> section.
langchain::embedding::HttpEmbeddingConfig get_embedding_config();

// Agent configuration from the <agent> section.
langchain::agent::AgentConfig get_agent_config();

// Memory configuration from the <memory> section.
struct MemoryConfig
{
    bool enabled = true;
    std::string backend = "sqlite";
    std::string db_path = "data/memory.db";
    std::string default_session_id = "default";
    int max_history_turns = 20;
};
MemoryConfig get_memory_config();

// RAG configuration from the <rag> section.
struct RagConfig
{
    bool enabled = true;
    std::string knowledge_dir = "data/knowledge";
    int retriever_top_k = 5;
    int chunk_size = 500;
    int chunk_overlap = 50;
};
RagConfig get_rag_config();

// Logging configuration from the <app> section.
struct LogConfig
{
    std::string level = "info";
    std::string dir = "logs";
    std::string filename = "smart_app.log";
    std::size_t max_size = 500 * 1024 * 1024;
    std::size_t max_files = 10;
    bool console = true;
};
LogConfig get_log_config();

// Tool configuration from the <tool> section.
struct ToolConfig
{
    bool calculator_enabled = true;
    bool http_get_enabled = true;
    bool datetime_enabled = true;
    bool echo_enabled = true;
};
ToolConfig get_tool_config();

// Skill configuration from the <skill> section.
struct SkillConfig
{
    bool enabled = true;
    std::string skills_dir = "skills";
};
SkillConfig get_skill_config();

} // namespace smart_app
