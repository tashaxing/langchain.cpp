// langchain/llm/qwen_llm.h
// Alibaba Cloud Qwen (DashScope) API backend.
// https://dashscope.aliyun.com/
#pragma once

#include "llm/llm.h"

#include <string>

namespace langchain
{
namespace llm
{

struct QwenLLMConfig
{
    std::string base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1";
    std::string api_key;                              // required
    std::string model    = "qwen-turbo";
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;
};

class QwenLLM : public ILLM
{
public:
    explicit QwenLLM(QwenLLMConfig cfg);
    ~QwenLLM() override = default;

    std::string name() const override
    {
        return "qwen:" + cfg_.model;
    }

    const QwenLLMConfig& config() const
    {
        return cfg_;
    }

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    json build_payload(const ChatRequest& req, bool stream) const;
    QwenLLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
