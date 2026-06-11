// examples/04_rag.cpp — minimal RAG with SQLite vector store + hashing embedder.
// Uses no network; demonstrates the wiring.
#include "langchain.h"

#include <filesystem>
#include <iostream>

int main()
{
    using namespace langchain;

    std::filesystem::create_directories("build");

    auto embedder = std::make_shared<embedding::HashingEmbedding>(256);
    vectorstore::SqliteVectorStore store(embedder, "build/rag_demo.db");

    if (store.size() == 0)
    {
        store.add_documents({
            {"", "C++20 introduces modules, coroutines, concepts, and ranges.", {}, {}},
            {"", "std::span is a non-owning view over a contiguous sequence.", {}, {}},
            {"", "std::jthread automatically joins on destruction.", {}, {}},
            {"", "Python's GIL prevents true thread-level parallelism for CPU-bound work.", {}, {}}
        });
    }

    auto hits = store.similarity_search("Tell me about C++ span", 2);
    std::cout << "Top hits:\n";
    for (const auto& h : hits)
    {
        std::cout << "  [" << h.score << "] " << h.doc.content << "\n";
    }
    return 0;
}
