// langchain/llm/deepseek_llm.h
// DeepSeek API backend.
// https://platform.deepseek.com/
#pragma once

#include "llm/llm.h"

#include <string>

namespace langchain
{
namespace llm
{

struct DeepSeekLLMConfig
{
    std::string base_url = "https://api.deepseek.com";
    std::string api_key;                              // required
    std::string model    = "deepseek-chat";
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;
};

class DeepSeekLLM : public ILLM
{
public:
    explicit DeepSeekLLM(DeepSeekLLMConfig cfg);
    ~DeepSeekLLM() override = default;

    std::string name() const override
    {
        return "deepseek:" + cfg_.model;
    }

    const DeepSeekLLMConfig& config() const
    {
        return cfg_;
    }

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    json build_payload(const ChatRequest& req, bool stream) const;
    DeepSeekLLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
