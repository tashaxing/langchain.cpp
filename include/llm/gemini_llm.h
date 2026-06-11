// langchain/llm/gemini_llm.h
// Google Gemini API backend.
// https://ai.google.dev/
#pragma once

#include "llm/llm.h"

#include <string>

namespace langchain
{
namespace llm
{

struct GeminiLLMConfig
{
    std::string base_url = "https://generativelanguage.googleapis.com/v1beta";
    std::string api_key;                              // required
    std::string model    = "gemini-1.5-flash-latest";
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;
};

class GeminiLLM : public ILLM
{
public:
    explicit GeminiLLM(GeminiLLMConfig cfg);
    ~GeminiLLM() override = default;

    std::string name() const override
    {
        return "gemini:" + cfg_.model;
    }

    const GeminiLLMConfig& config() const
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
    GeminiLLMConfig cfg_;
};

} // namespace llm
} // namespace langchain
