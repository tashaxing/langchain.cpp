// langchain/retriever/retriever.h
// Retriever abstraction: decouples "how to retrieve" from the concrete
// vector store implementation.
#pragma once

#include "vectorstore/vectorstore.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace langchain
{
namespace retriever
{

// ---------------------------------------------------------------------------
// BaseRetriever -- abstract retrieval interface.
// ---------------------------------------------------------------------------
class BaseRetriever
{
public:
    virtual ~BaseRetriever() = default;

    // Retrieve top-k documents for a given query.
    virtual std::vector<vectorstore::ScoredDocument> retrieve(
        const std::string& query, int k = 4) = 0;
};

using RetrieverPtr = std::shared_ptr<BaseRetriever>;

// ---------------------------------------------------------------------------
// VectorStoreRetriever -- thin wrapper around any IVectorStore.
// ---------------------------------------------------------------------------
class VectorStoreRetriever : public BaseRetriever
{
public:
    VectorStoreRetriever(vectorstore::VectorStorePtr vs, int default_k = 4);

    std::vector<vectorstore::ScoredDocument> retrieve(
        const std::string& query, int k = 4) override;

private:
    vectorstore::VectorStorePtr vs_;
    int default_k_;
};

// ---------------------------------------------------------------------------
// BM25Retriever -- keyword-based sparse retrieval.
// ---------------------------------------------------------------------------
class BM25Retriever : public BaseRetriever
{
public:
    BM25Retriever(const std::vector<Document>& docs,
                  float k1 = 1.5f,
                  float b = 0.75f);
    ~BM25Retriever();

    std::vector<vectorstore::ScoredDocument> retrieve(
        const std::string& query, int k = 4) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// MultiQueryRetriever -- expands query into variants and merges results.
// ---------------------------------------------------------------------------
class MultiQueryRetriever : public BaseRetriever
{
public:
    // expander: function that generates query variants.
    MultiQueryRetriever(RetrieverPtr base,
                        std::vector<std::string> (*expander)(const std::string&));

    std::vector<vectorstore::ScoredDocument> retrieve(
        const std::string& query, int k = 4) override;

private:
    RetrieverPtr base_;
    std::vector<std::string> (*expander_)(const std::string&);
};

} // namespace retriever
} // namespace langchain
