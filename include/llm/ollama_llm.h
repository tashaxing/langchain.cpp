// langchain/llm/ollama_llm.h
// Ollama local API backend.
// https://ollama.com/
#pragma once

#include "llm/llm.h"

#include <string>
#include <vector>

namespace langchain
{
namespace llm
{

struct OllamaLLMConfig
{
    std::string base_url = "http://localhost:11434";
    std::string model    = "llama3.2";
    std::string api_key;  // optional (for reverse proxy setups)
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 300; // local models can be slow to start
};

class OllamaLLM : public ILLM
{
public:
    explicit OllamaLLM(OllamaLLMConfig cfg);
    ~OllamaLLM() override = default;

    std::string name() const override
    {
        return "ollama:" + cfg_.model;
    }

    const OllamaLLMConfig& config() const
    {
        return cfg_;
    }

    // Ollama-specific: list local models
    std::vector<std::string> list_models() const;

    // Ollama-specific: pull a model
    void pull_model(const std::string& model) const;

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    json build_payload(const ChatRequest& req, bool stream) const;
    Message decode_response(const json& j) const;
    OllamaLLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
