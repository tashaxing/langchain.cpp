// examples/19_bm25_retriever.cpp — Retrieval strategies: BM25 + MultiQuery.
//
// Demonstrates:
//   1. BM25Retriever: keyword-based sparse retrieval (no embeddings).
//   2. MultiQueryRetriever: expand query into variants, merge results.
//   3. VectorStoreRetriever: dense similarity search wrapper.
//
// No network required.
#include "langchain.h"

// BM25Retriever::Impl is defined in retriever.cpp; we only use the class
// through its public interface, so the header is sufficient.
#include <iostream>
#include <vector>

int main()
{
    using namespace langchain;

    // ---- Sample documents ----
    std::vector<Document> docs = {
        {"doc1", "C++20 introduces modules, coroutines, and concepts.", {}, {}},
        {"doc2", "Python has a Global Interpreter Lock that limits threading.", {}, {}},
        {"doc3", "Rust ownership system prevents data races at compile time.", {}, {}},
        {"doc4", "C++ modules replace header files and improve build times.", {}, {}},
        {"doc5", "Python asyncio provides cooperative multitasking.", {}, {}},
    };

    // ---- 1. BM25Retriever ----
    std::cout << "=== BM25Retriever ===\n";
    retriever::BM25Retriever bm25(docs);
    auto bm25_hits = bm25.retrieve("C++ modules", 3);
    std::cout << "Query: 'C++ modules'\n";
    for (const auto& h : bm25_hits)
    {
        std::cout << "  [score=" << h.score << "] " << h.doc.content << "\n";
    }

    // ---- 2. VectorStoreRetriever ----
    std::cout << "\n=== VectorStoreRetriever ===\n";
    auto embedder = std::make_shared<embedding::HashingEmbedding>(128);
    auto vs = std::make_shared<vectorstore::InMemoryVectorStore>(embedder);
    vs->add_documents(docs);

    retriever::VectorStoreRetriever dense(vs, 3);
    auto dense_hits = dense.retrieve("C++ modules");
    std::cout << "Query: 'C++ modules'\n";
    for (const auto& h : dense_hits)
    {
        std::cout << "  [score=" << h.score << "] " << h.doc.content << "\n";
    }

    // ---- 3. MultiQueryRetriever ----
    std::cout << "\n=== MultiQueryRetriever ===\n";
    auto expander = [](const std::string& q) -> std::vector<std::string>
    {
        return {
            q,
            q + " tutorial",
            q + " examples"
        };
    };

    retriever::MultiQueryRetriever multi(
        std::make_shared<retriever::VectorStoreRetriever>(dense),
        expander);

    auto multi_hits = multi.retrieve("C++", 3);
    std::cout << "Query: 'C++' (expanded to 3 variants)\n";
    for (const auto& h : multi_hits)
    {
        std::cout << "  [score=" << h.score << "] " << h.doc.content << "\n";
    }

    return 0;
}
