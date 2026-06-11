// examples/22_vectorstore_crud.cpp — VectorStore CRUD operations.
//
// Demonstrates:
//   1. InMemoryVectorStore: add, search, delete, update, filter.
//   2. SqliteVectorStore: persistence across process restarts.
//   3. Metadata filtering in similarity search.
//
// No network required.
#include "langchain.h"

#include <filesystem>
#include <iostream>

int main()
{
    using namespace langchain;

    auto embedder = std::make_shared<embedding::HashingEmbedding>(128);

    // ---- 1. InMemoryVectorStore CRUD ----
    std::cout << "=== InMemoryVectorStore ===\n";
    vectorstore::InMemoryVectorStore mem_vs(embedder);

    // Add documents with IDs and metadata
    auto ids = mem_vs.add_documents({
        {"cpp20", "C++20 introduces modules and coroutines.",
         {{"category", "cpp"}, {"year", "2020"}}, {}},
        {"cpp23", "C++23 adds std::expected and std::print.",
         {{"category", "cpp"}, {"year", "2023"}}, {}},
        {"py311", "Python 3.11 improves asyncio performance.",
         {{"category", "python"}, {"year", "2021"}}, {}},
    });
    std::cout << "Added " << ids.size() << " documents\n";

    // Search all
    auto hits = mem_vs.similarity_search("C++ features", 10);
    std::cout << "\nSearch 'C++ features':\n";
    for (const auto& h : hits)
    {
        std::cout << "  [" << h.doc.id << ", score=" << h.score << "] "
                  << h.doc.content << "\n";
    }

    // Filter by metadata
    auto filtered = mem_vs.similarity_search(
        "programming", 10, {{"category", "cpp"}});
    std::cout << "\nFiltered by category=cpp:\n";
    for (const auto& h : filtered)
    {
        std::cout << "  [" << h.doc.id << ", score=" << h.score << "] "
                  << h.doc.content << "\n";
    }

    // Update a document
    std::cout << "\nUpdating doc 'cpp20'...\n";
    Document updated;
    updated.content = "C++20 introduces modules, coroutines, concepts, and ranges.";
    updated.metadata = {{"category", "cpp"}, {"year", "2020"}, {"status", "updated"}};
    mem_vs.update_document("cpp20", std::move(updated));

    auto after_update = mem_vs.similarity_search("C++20 ranges", 1);
    if (!after_update.empty())
    {
        std::cout << "After update: " << after_update[0].doc.content << "\n";
    }

    // Delete a document
    std::cout << "\nDeleting doc 'py311'...\n";
    std::size_t removed = mem_vs.delete_documents({"py311"});
    std::cout << "Removed: " << removed << ", size now: " << mem_vs.size() << "\n";

    // ---- 2. SqliteVectorStore persistence ----
    std::cout << "\n=== SqliteVectorStore (persistence) ===\n";
    std::filesystem::create_directories("build");
    std::string db_path = "build/vectorstore_crud_demo.db";
    std::filesystem::remove(db_path);

    // Phase 1: create and populate
    {
        vectorstore::SqliteVectorStore sqlite_vs(embedder, db_path);
        sqlite_vs.add_documents({
            {"persist1", "This document will survive process restart.",
             {{"tag", "persistent"}}, {}},
            {"persist2", "Another persistent document.",
             {{"tag", "persistent"}}, {}},
        });
        std::cout << "Phase 1: stored " << sqlite_vs.size() << " documents\n";
    }

    // Phase 2: reopen and verify
    {
        vectorstore::SqliteVectorStore sqlite_vs(embedder, db_path);
        std::cout << "Phase 2: loaded " << sqlite_vs.size() << " documents\n";

        auto hits = sqlite_vs.similarity_search("survive restart", 2);
        std::cout << "Search results:\n";
        for (const auto& h : hits)
        {
            std::cout << "  [" << h.doc.id << ", score=" << h.score << "] "
                      << h.doc.content << "\n";
        }

        // Delete persists too
        sqlite_vs.delete_documents({"persist1"});
        std::cout << "After delete: " << sqlite_vs.size() << " documents\n";
    }

    // Phase 3: verify deletion persisted
    {
        vectorstore::SqliteVectorStore sqlite_vs(embedder, db_path);
        std::cout << "Phase 3: loaded " << sqlite_vs.size()
                  << " documents (delete persisted)\n";
    }

    std::filesystem::remove(db_path);
    std::cout << "\nDone.\n";
    return 0;
}
