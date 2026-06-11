// langchain/llm/anthropic_llm.h
// Anthropic Claude API backend.
// https://www.anthropic.com/
#pragma once

#include "llm/llm.h"

#include <string>

namespace langchain
{
namespace llm
{

struct AnthropicLLMConfig
{
    std::string base_url = "https://api.anthropic.com";
    std::string api_key;                              // required
    std::string model    = "claude-3-5-sonnet-20241022";
    int max_tokens       = 1024;                      // required for Anthropic
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;
};

class AnthropicLLM : public ILLM
{
public:
    explicit AnthropicLLM(AnthropicLLMConfig cfg);
    ~AnthropicLLM() override = default;

    std::string name() const override
    {
        return "anthropic:" + cfg_.model;
    }

    const AnthropicLLMConfig& config() const
    {
        return cfg_;
    }

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    json build_payload(const ChatRequest& req, bool stream) const;
    Message decode_response(const json& j) const;
    AnthropicLLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
