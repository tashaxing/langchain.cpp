// langchain/embedding/llamacpp_embedding.h
// Local text embedding using a bundled llama.cpp model.
//
// This backend loads a GGUF model once and runs forward passes to extract
// the final hidden state as the embedding vector.  It is only available when
// the project is built with -DLC_ENABLE_LLAMA=ON.
//
// Usage:
//   embedding::LlamaCppEmbeddingConfig cfg;
//   cfg.model_path = "models/all-MiniLM-L6-v2.Q4_K_M.gguf";
//   cfg.dimension  = 384;
//   auto embedder = std::make_shared<embedding::LlamaCppEmbedding>(cfg);
//   auto vec = embedder->embed_query("hello world");
//
// Model recommendations:
//   - sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)
//   - nomic-ai/nomic-embed-text-v1.5 (768-dim, high quality)
//   - BAAI/bge-small-en-v1.5 (384-dim, good for retrieval)
//
// These models can be downloaded as GGUF from HuggingFace and must be
// explicitly converted if only the PyTorch/Safetensors variant exists.
#pragma once

#include "embedding/embedding.h"

#include <string>

namespace langchain
{
namespace embedding
{

struct LlamaCppEmbeddingConfig
{
    // Path to the GGUF file.
    std::string model_path;

    // Embedding dimension (must match the model).
    int dimension = 384;

    // Context size for the internal llama_context.
    int n_ctx = 512;

    // Number of GPU layers to offload (0 = CPU only).
    int n_gpu_layers = 0;

    // Number of threads for inference (0 = auto).
    int n_threads = 0;

    // Whether to prepend a pooling/normalisation step.
    // For many embedding models the raw hidden state is already the desired
    // vector, so this defaults to false.
    bool normalize = false;
};

// ---------------------------------------------------------------------------
// LlamaCppEmbedding
// ---------------------------------------------------------------------------
// Only compiled when LC_ENABLE_LLAMA=ON.  When llama is disabled the
// constructor throws LCError and all other methods return empty vectors.
// ---------------------------------------------------------------------------
class LlamaCppEmbedding : public IEmbedding
{
public:
    explicit LlamaCppEmbedding(LlamaCppEmbeddingConfig cfg);
    ~LlamaCppEmbedding() override;

    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override;

    int dimension() const override
    {
        return cfg_.dimension;
    }

    std::string name() const override
    {
        return "llamacpp-embed:" + cfg_.model_path;
    }

private:
    struct Impl;
    LlamaCppEmbeddingConfig cfg_;
    Impl* impl_ = nullptr;

    std::vector<float> embed_single_(const std::string& text);
};

} // namespace embedding
} // namespace langchain
