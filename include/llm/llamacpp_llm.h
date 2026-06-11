// langchain/llm/llamacpp_llm.h
// llama.cpp in-process backend. Built only when LC_ENABLE_LLAMA=ON.
#pragma once

#include "llm/llm.h"

#include <memory>
#include <string>

namespace langchain
{
namespace llm
{

struct LlamacppConfig
{
    std::string model_path; // path to GGUF
    int n_ctx        = 4096;
    int n_gpu_layers = 0;
    int n_threads    = 0; // 0 => auto
    int seed         = -1;
};

class LlamacppLLM : public ILLM
{
public:
    explicit LlamacppLLM(LlamacppConfig cfg);
    ~LlamacppLLM() override;

    LlamacppLLM(const LlamacppLLM&)            = delete;
    LlamacppLLM& operator=(const LlamacppLLM&) = delete;

    std::string name() const override
    {
        return "llama.cpp:" + cfg_.model_path;
    }

protected:
    ChatResponse invoke_impl(const ChatRequest& req) override;
    ChatResponse invoke_stream_impl(const ChatRequest& req,
                                  const StreamCallback& on_delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    LlamacppConfig cfg_;
};

} // namespace llm
} // namespace langchain
