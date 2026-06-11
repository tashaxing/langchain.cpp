// langchain/llm/openai_common.h
// Shared helpers for OpenAI-compatible backends (OpenAI, DeepSeek, Groq, Qwen).
#pragma once

#include <string>

namespace langchain
{
namespace llm
{

// Build the chat completions path, avoiding double /v1 when base_url already
// contains a trailing /v1 (or /v1/...) path segment.
//
// Examples:
//   "http://host"               -> "http://host/v1/chat/completions"
//   "http://host/llm-service"   -> "http://host/llm-service/v1/chat/completions"
//   "http://host/llm-service/v1" -> "http://host/llm-service/v1/chat/completions"
inline std::string make_chat_completions_path(const std::string& path_prefix)
{
    if (!path_prefix.empty())
    {
        if (path_prefix.size() >= 3 &&
            path_prefix.compare(path_prefix.size() - 3, 3, "/v1") == 0)
        {
            return path_prefix + "/chat/completions";
        }
        if (path_prefix.find("/v1/") != std::string::npos)
        {
            return path_prefix + "/chat/completions";
        }
    }
    return path_prefix + "/v1/chat/completions";
}

} // namespace llm
} // namespace langchain
