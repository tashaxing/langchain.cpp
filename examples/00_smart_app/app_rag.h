// app_rag.h -- RAG pipeline setup.
#pragma once

#include "langchain.h"

#include <memory>
#include <string>

namespace smart_app
{

// Build and return the vector store (SQLite-backed with HTTP embedding).
std::shared_ptr<langchain::vectorstore::IVectorStore> build_vectorstore();

// Ingest all .txt and .md files from a directory into the vector store.
// Returns the number of chunks ingested.
std::size_t ingest_documents(const std::string& dir);

// Build a retrieval skill backed by the vector store.
langchain::skill::RetrievalSkill build_retrieval_skill(
    langchain::llm::LLMPtr llm,
    std::shared_ptr<langchain::vectorstore::IVectorStore> store);

} // namespace smart_app
