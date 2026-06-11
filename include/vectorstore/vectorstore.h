// langchain/vectorstore/vectorstore.h
// IVectorStore + multiple implementations.
#pragma once

#include "embedding/embedding.h"
#include "util/common.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace langchain
{
namespace vectorstore
{

struct ScoredDocument
{
    Document doc;
    float score = 0.0f; // higher == more similar (cosine)
};

class IVectorStore
{
public:
    virtual ~IVectorStore() = default;

    // Embed and persist a batch of documents. Returns the assigned ids.
    virtual std::vector<std::string> add_documents(std::vector<Document> docs) = 0;

    // Cosine top-k against the embedded query.
    // `filter` is an optional key-value map; only documents whose metadata
    // contain ALL specified pairs are considered.
    virtual std::vector<ScoredDocument> similarity_search(
        const std::string& query, int k = 4,
        const std::unordered_map<std::string, std::string>& filter = {}) = 0;

    // Delete documents by id. Returns number actually removed.
    virtual std::size_t delete_documents(const std::vector<std::string>& ids) = 0;

    // Update a single document (replace content + metadata, re-embed).
    virtual void update_document(const std::string& id, Document doc) = 0;

    virtual std::size_t size() const = 0;
};

using VectorStorePtr = std::shared_ptr<IVectorStore>;

// Brute-force cosine over a std::vector. Good up to ~10^5 docs.
class InMemoryVectorStore : public IVectorStore
{
public:
    explicit InMemoryVectorStore(embedding::EmbeddingPtr embedder)
        : embedder_(std::move(embedder))
    {
    }

    std::vector<std::string> add_documents(std::vector<Document> docs) override;
    std::vector<ScoredDocument> similarity_search(
        const std::string& query, int k = 4,
        const std::unordered_map<std::string, std::string>& filter = {}) override;
    std::size_t delete_documents(const std::vector<std::string>& ids) override;
    void update_document(const std::string& id, Document doc) override;
    std::size_t size() const override;

protected:
    embedding::EmbeddingPtr embedder_;
    std::vector<Document> docs_;
    mutable std::mutex mu_;
};

// SQLite persistence; in-process cosine ranking happens in C++ after loading
// candidates. Suitable for small/medium corpora.
class SqliteVectorStore : public InMemoryVectorStore
{
public:
    SqliteVectorStore(embedding::EmbeddingPtr embedder,
                      const std::string& db_path);
    ~SqliteVectorStore() override;

    std::vector<std::string> add_documents(std::vector<Document> docs) override;
    std::size_t delete_documents(const std::vector<std::string>& ids) override;
    void update_document(const std::string& id, Document doc) override;

private:
    void open_(const std::string& path);
    void load_all_();
    void persist_delete_(const std::string& id);
    void persist_update_(const Document& doc);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// FAISS-backed vector store with IVF or Flat index. Good for 10^5+ docs.
class FaissVectorStore : public IVectorStore
{
public:
    // If nlist == 0, uses IndexFlatIP (exact search). Otherwise IndexIVFFlat.
    FaissVectorStore(embedding::EmbeddingPtr embedder,
                     int nlist = 0,
                     int nprobe = 1);
    ~FaissVectorStore() override;

    std::vector<std::string> add_documents(std::vector<Document> docs) override;
    std::vector<ScoredDocument> similarity_search(
        const std::string& query, int k = 4,
        const std::unordered_map<std::string, std::string>& filter = {}) override;
    std::size_t delete_documents(const std::vector<std::string>& ids) override;
    void update_document(const std::string& id, Document doc) override;
    std::size_t size() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    embedding::EmbeddingPtr embedder_;
    std::vector<Document> docs_;
    mutable std::mutex mu_;
};

} // namespace vectorstore
} // namespace langchain
