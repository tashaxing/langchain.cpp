// src/embedding/llamacpp_embedding.cpp — llama.cpp-based local embedding.
// Only compiled when LC_ENABLE_LLAMA=ON (gated by CMakeLists.txt).
#include "embedding/llamacpp_embedding.h"

#if defined(LC_HAS_LLAMA)
#include <llama.h>
#endif

#include "util/logging.h"

#include <cmath>
#include <cstring>
#include <sstream>

namespace langchain
{
namespace embedding
{

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct LlamaCppEmbedding::Impl
{
#if defined(LC_HAS_LLAMA)
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    const llama_vocab* vocab = nullptr;
#endif
};

// ---------------------------------------------------------------------------
// Helpers (llama-enabled only)
// ---------------------------------------------------------------------------

#if defined(LC_HAS_LLAMA)

namespace
{

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                    const std::string& text,
                                    bool add_special)
{
    int32_t n_neg = -llama_tokenize(vocab, text.c_str(),
                                    static_cast<int32_t>(text.size()),
                                    nullptr, 0, add_special,
                                    /*parse_special=*/true);
    std::vector<llama_token> out(static_cast<std::size_t>(n_neg));
    int32_t n = llama_tokenize(vocab, text.c_str(),
                               static_cast<int32_t>(text.size()),
                               out.data(), static_cast<int32_t>(out.size()),
                               add_special, /*parse_special=*/true);
    if (n < 0)
    {
        throw LCError("LlamaCppEmbedding: tokenize failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return out;
}

// Mean-pool the token embeddings into a single sentence vector.
// embeddings: flat array of n_tokens * n_embd floats.
std::vector<float> mean_pool(const float* embeddings,
                               std::size_t n_tokens,
                               int n_embd)
{
    std::vector<float> pooled(static_cast<std::size_t>(n_embd), 0.0f);
    for (std::size_t t = 0; t < n_tokens; ++t)
    {
        for (int i = 0; i < n_embd; ++i)
        {
            pooled[i] += embeddings[t * n_embd + i];
        }
    }
    float inv = 1.0f / static_cast<float>(n_tokens);
    for (float& v : pooled)
    {
        v *= inv;
    }
    return pooled;
}

void l2_normalize(std::vector<float>& vec)
{
    double norm = 0.0;
    for (float v : vec)
    {
        norm += static_cast<double>(v) * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0)
    {
        for (float& v : vec)
        {
            v = static_cast<float>(v / norm);
        }
    }
}

} // namespace

#endif // LC_HAS_LLAMA

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

LlamaCppEmbedding::LlamaCppEmbedding(LlamaCppEmbeddingConfig cfg)
    : cfg_(std::move(cfg))
{
    impl_ = new Impl();

#if defined(LC_HAS_LLAMA)
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg_.n_gpu_layers;

    impl_->model = llama_model_load_from_file(cfg_.model_path.c_str(), mparams);
    if (!impl_->model)
    {
        delete impl_;
        impl_ = nullptr;
        throw LCError("LlamaCppEmbedding: failed to load model: " + cfg_.model_path);
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = cfg_.n_ctx;
    cparams.n_batch = cfg_.n_ctx;
    cparams.embeddings = true; // required for embedding extraction
    if (cfg_.n_threads > 0)
    {
        cparams.n_threads       = cfg_.n_threads;
        cparams.n_threads_batch = cfg_.n_threads;
    }

    impl_->ctx = llama_init_from_model(impl_->model, cparams);
    if (!impl_->ctx)
    {
        llama_model_free(impl_->model);
        delete impl_;
        impl_ = nullptr;
        throw LCError("LlamaCppEmbedding: failed to create context");
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);

    // Validate dimension against the model's actual n_embd.
    int model_embd = llama_model_n_embd(impl_->model);
    if (cfg_.dimension != model_embd)
    {
        LOG_WARN("LlamaCppEmbedding: configured dimension ({}) does not match "
                 "model n_embd ({}).  Using model value.",
                 cfg_.dimension, model_embd);
        cfg_.dimension = model_embd;
    }
#else
    (void)cfg_;
    throw LCError(
        "LlamaCppEmbedding: llama.cpp backend not compiled in. "
        "Rebuild with -DLC_ENABLE_LLAMA=ON and ensure deps/llama.cpp-b5018 is present.");
#endif
}

LlamaCppEmbedding::~LlamaCppEmbedding()
{
#if defined(LC_HAS_LLAMA)
    if (impl_)
    {
        if (impl_->ctx)
        {
            llama_free(impl_->ctx);
        }
        if (impl_->model)
        {
            llama_model_free(impl_->model);
        }
        llama_backend_free();
    }
#endif
    delete impl_;
}

// ---------------------------------------------------------------------------
// embed_documents
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> LlamaCppEmbedding::embed_documents(
    const std::vector<std::string>& texts)
{
    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& t : texts)
    {
        out.push_back(embed_single_(t));
    }
    return out;
}

// ---------------------------------------------------------------------------
// embed_single_
// ---------------------------------------------------------------------------

std::vector<float> LlamaCppEmbedding::embed_single_(const std::string& text)
{
#if defined(LC_HAS_LLAMA)
    if (!impl_ || !impl_->ctx || !impl_->model)
    {
        throw LCError("LlamaCppEmbedding: not initialized");
    }

    auto tokens = tokenize(impl_->vocab, text, /*add_special=*/true);
    if (tokens.empty())
    {
        return std::vector<float>(static_cast<std::size_t>(cfg_.dimension), 0.0f);
    }

    if (static_cast<int>(tokens.size()) > cfg_.n_ctx)
    {
        LOG_WARN("LlamaCppEmbedding: input token count ({}) exceeds n_ctx ({}). "
                 "Truncating.", tokens.size(), cfg_.n_ctx);
        tokens.resize(static_cast<std::size_t>(cfg_.n_ctx));
    }

    llama_batch batch = llama_batch_get_one(
        tokens.data(), static_cast<int32_t>(tokens.size()));

    int decode_result = llama_decode(impl_->ctx, batch);
    if (decode_result != 0)
    {
        if (decode_result == 1)
        {
            throw LCError("LlamaCppEmbedding: could not find a KV slot for the batch "
                          "(try increasing n_ctx)");
        }
        throw LCError("LlamaCppEmbedding: llama_decode failed");
    }

    int n_embd = llama_model_n_embd(impl_->model);
    std::size_t n_tokens = tokens.size();

    // llama_get_embeddings returns a flat array of n_tokens * n_embd.
    const float* emb = llama_get_embeddings(impl_->ctx);
    if (!emb)
    {
        throw LCError("LlamaCppEmbedding: llama_get_embeddings returned null");
    }

    std::vector<float> vec = mean_pool(emb, n_tokens, n_embd);

    if (cfg_.normalize)
    {
        l2_normalize(vec);
    }

    return vec;
#else
    (void)text;
    return {};
#endif
}

} // namespace embedding
} // namespace langchain
