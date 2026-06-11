// langchain/llm/groq_llm.h
// Groq API backend (OpenAI-compatible, ultra-fast inference).
// https://groq.com/
#pragma once

#include "llm/llm.h"

#include <string>

namespace langchain
{
namespace llm
{

struct GroqLLMConfig
{
    std::string base_url = "https://api.groq.com/openai/v1";
    std::string api_key;                              // required
    std::string model    = "llama-3.1-8b-instant";
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;
};

class GroqLLM : public ILLM
{
public:
    explicit GroqLLM(GroqLLMConfig cfg);
    ~GroqLLM() override = default;

    std::string name() const override
    {
        return "groq:" + cfg_.model;
    }

    const GroqLLMConfig& config() const
    {
        return cfg_;
    }

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    json build_payload(const ChatRequest& req, bool stream) const;
    GroqLLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
