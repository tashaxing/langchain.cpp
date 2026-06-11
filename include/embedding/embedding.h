// langchain/embedding/embedding.h
// IEmbedding interface + multiple implementations.
#pragma once

#include "util/common.h"

#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace embedding
{

class IEmbedding
{
public:
    virtual ~IEmbedding() = default;

    // Batch embed; one float-vector per input document.
    virtual std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) = 0;

    // Single embedding; default routes through embed_documents.
    virtual std::vector<float> embed_query(const std::string& text)
    {
        auto v = embed_documents({text});
        return v.empty() ? std::vector<float>{} : std::move(v[0]);
    }

    virtual int dimension() const = 0;
    virtual std::string name() const = 0;
};

using EmbeddingPtr = std::shared_ptr<IEmbedding>;

// ---------------------------------------------------------------------------
// HttpEmbedding -- OpenAI-compatible /v1/embeddings endpoint.
// ---------------------------------------------------------------------------
struct HttpEmbeddingConfig
{
    std::string base_url = "https://api.openai.com";
    std::string path     = "/v1/embeddings";
    std::string api_key;
    std::string model    = "text-embedding-3-small";
    int dimension        = 1536;
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 60;
};

class HttpEmbedding : public IEmbedding
{
public:
    explicit HttpEmbedding(HttpEmbeddingConfig cfg)
        : cfg_(std::move(cfg))
    {
    }

    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override;

    int dimension() const override
    {
        return cfg_.dimension;
    }
    std::string name() const override
    {
        return "http-embed:" + cfg_.model;
    }

private:
    HttpEmbeddingConfig cfg_;
};

// ---------------------------------------------------------------------------
// OllamaEmbedding -- calls a local Ollama /api/embeddings endpoint.
// ---------------------------------------------------------------------------
struct OllamaEmbeddingConfig
{
    std::string base_url = "http://localhost:11434";
    std::string model    = "nomic-embed-text";
    int dimension        = 768;
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 60;
};

class OllamaEmbedding : public IEmbedding
{
public:
    explicit OllamaEmbedding(OllamaEmbeddingConfig cfg = {})
        : cfg_(std::move(cfg))
    {
    }

    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override;

    int dimension() const override
    {
        return cfg_.dimension;
    }
    std::string name() const override
    {
        return "ollama-embed:" + cfg_.model;
    }

private:
    OllamaEmbeddingConfig cfg_;
};

// ---------------------------------------------------------------------------
// LocalAIEmbedding -- OpenAI-compatible local API (LocalAI, LM Studio, etc.).
// ---------------------------------------------------------------------------
struct LocalAIEmbeddingConfig
{
    std::string base_url = "http://localhost:8080";
    std::string path     = "/v1/embeddings";
    std::string api_key;
    std::string model    = "text-embedding-ada-002";
    int dimension        = 1536;
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 60;
};

class LocalAIEmbedding : public IEmbedding
{
public:
    explicit LocalAIEmbedding(LocalAIEmbeddingConfig cfg = {})
        : cfg_(std::move(cfg))
    {
    }

    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override;

    int dimension() const override
    {
        return cfg_.dimension;
    }
    std::string name() const override
    {
        return "localai-embed:" + cfg_.model;
    }

private:
    LocalAIEmbeddingConfig cfg_;
};

// ---------------------------------------------------------------------------
// HashingEmbedding -- deterministic, dependency-free for tests/offline demos.
// ---------------------------------------------------------------------------
class HashingEmbedding : public IEmbedding
{
public:
    explicit HashingEmbedding(int dim = 128)
        : dim_(dim)
    {
    }

    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override;

    int dimension() const override
    {
        return dim_;
    }
    std::string name() const override
    {
        return "hashing-embedding";
    }

private:
    int dim_;
};

} // namespace embedding
} // namespace langchain
