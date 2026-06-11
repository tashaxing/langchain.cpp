// langchain/llm/openai_llm.h
// OpenAI API backend (GPT-4, GPT-4o, GPT-3.5, etc.).
// Also works for any OpenAI-compatible private deployment.
// https://platform.openai.com/
#pragma once

#include "llm/llm.h"

#include <string>

namespace langchain
{
namespace llm
{

struct OpenAILLMConfig
{
    std::string base_url = "https://api.openai.com";
    std::string api_key;                              // required
    std::string model    = "gpt-4o-mini";
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;
};

class OpenAILLM : public ILLM
{
public:
    explicit OpenAILLM(OpenAILLMConfig cfg);
    ~OpenAILLM() override = default;

    std::string name() const override
    {
        return "openai:" + cfg_.model;
    }

    const OpenAILLMConfig& config() const
    {
        return cfg_;
    }

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    json build_payload(const ChatRequest& req, bool stream) const;
    OpenAILLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
