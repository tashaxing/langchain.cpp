// tests/test_llamacpp_embedding.cpp — LlamaCppEmbedding unit tests.
//
// These tests are compiled unconditionally but skipped at runtime when
// llama.cpp is not available (no LC_HAS_LLAMA).  This keeps the test
// binary buildable on all configurations.
#include <gtest/gtest.h>

#include "embedding/llamacpp_embedding.h"

#include <fstream>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool llama_available()
{
#ifdef LC_HAS_LLAMA
    return true;
#else
    return false;
#endif
}

// Check whether a file exists (no fs::exists to avoid GCC <9 link).
bool file_exists(const std::string& path)
{
    std::ifstream ifs(path);
    return ifs.good();
}

// Return a path to a small GGUF model if one is present in the environment.
std::string model_path_or_skip()
{
    const char* env = std::getenv("LC_TEST_EMBED_MODEL");
    if (env && file_exists(env))
    {
        return env;
    }
    // Common fallback paths checked by CI / local setups.
    const char* candidates[] = {
        "models/all-MiniLM-L6-v2.Q4_K_M.gguf",
        "models/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "../models/all-MiniLM-L6-v2.Q4_K_M.gguf",
    };
    for (const char* p : candidates)
    {
        if (file_exists(p))
        {
            return p;
        }
    }
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(LlamaCppEmbedding, ThrowsWhenLlamaNotCompiled)
{
    if (llama_available())
    {
        GTEST_SKIP() << "llama is compiled in — skipping negative test";
    }

    embedding::LlamaCppEmbeddingConfig cfg;
    cfg.model_path = "dummy.gguf";
    EXPECT_THROW((embedding::LlamaCppEmbedding(cfg)), LCError);
}

TEST(LlamaCppEmbedding, ThrowsWhenModelMissing)
{
    if (!llama_available())
    {
        GTEST_SKIP() << "llama not compiled in";
    }

    embedding::LlamaCppEmbeddingConfig cfg;
    cfg.model_path = "/nonexistent/model.gguf";
    EXPECT_THROW((embedding::LlamaCppEmbedding(cfg)), LCError);
}

// ---------------------------------------------------------------------------
// Embedding shape
// ---------------------------------------------------------------------------

TEST(LlamaCppEmbedding, EmbedQueryReturnsCorrectDimension)
{
    if (!llama_available())
    {
        GTEST_SKIP() << "llama not compiled in";
    }

    std::string path = model_path_or_skip();
    if (path.empty())
    {
        GTEST_SKIP() << "no test model found; set LC_TEST_EMBED_MODEL";
    }

    embedding::LlamaCppEmbeddingConfig cfg;
    cfg.model_path = path;
    cfg.dimension  = 384; // will be overridden by model n_embd

    embedding::LlamaCppEmbedding embedder(cfg);

    auto vec = embedder.embed_query("hello world");
    EXPECT_EQ(static_cast<int>(vec.size()), embedder.dimension());
    EXPECT_GT(vec.size(), 0u);
}

TEST(LlamaCppEmbedding, EmbedDocumentsBatch)
{
    if (!llama_available())
    {
        GTEST_SKIP() << "llama not compiled in";
    }

    std::string path = model_path_or_skip();
    if (path.empty())
    {
        GTEST_SKIP() << "no test model found; set LC_TEST_EMBED_MODEL";
    }

    embedding::LlamaCppEmbeddingConfig cfg;
    cfg.model_path = path;
    embedding::LlamaCppEmbedding embedder(cfg);

    auto vecs = embedder.embed_documents({"first sentence", "second sentence"});
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs[0].size(), static_cast<std::size_t>(embedder.dimension()));
    EXPECT_EQ(vecs[1].size(), static_cast<std::size_t>(embedder.dimension()));
}

// ---------------------------------------------------------------------------
// Semantic consistency (weak — no ground truth, just shape checks)
// ---------------------------------------------------------------------------

TEST(LlamaCppEmbedding, SameTextProducesSameVector)
{
    if (!llama_available())
    {
        GTEST_SKIP() << "llama not compiled in";
    }

    std::string path = model_path_or_skip();
    if (path.empty())
    {
        GTEST_SKIP() << "no test model found";
    }

    embedding::LlamaCppEmbeddingConfig cfg;
    cfg.model_path = path;
    embedding::LlamaCppEmbedding embedder(cfg);

    auto a = embedder.embed_query("the quick brown fox");
    auto b = embedder.embed_query("the quick brown fox");

    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}

TEST(LlamaCppEmbedding, EmptyTextReturnsZeroVector)
{
    if (!llama_available())
    {
        GTEST_SKIP() << "llama not compiled in";
    }

    std::string path = model_path_or_skip();
    if (path.empty())
    {
        GTEST_SKIP() << "no test model found";
    }

    embedding::LlamaCppEmbeddingConfig cfg;
    cfg.model_path = path;
    embedding::LlamaCppEmbedding embedder(cfg);

    auto vec = embedder.embed_query("");
    EXPECT_EQ(vec.size(), static_cast<std::size_t>(embedder.dimension()));
    for (float v : vec)
    {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}
