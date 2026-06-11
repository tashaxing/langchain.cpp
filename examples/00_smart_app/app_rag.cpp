// app_rag.cpp -- RAG pipeline: loader, splitter, embedder, vectorstore.
#include "app_rag.h"
#include "app_config.h"

#include "util/logging.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace smart_app
{

namespace
{

std::shared_ptr<langchain::vectorstore::IVectorStore> g_store;

} // namespace

std::shared_ptr<langchain::vectorstore::IVectorStore> build_vectorstore()
{
    using namespace langchain;

    RagConfig rcfg = get_rag_config();
    MemoryConfig mcfg = get_memory_config();

    if (!rcfg.enabled)
    {
        LOG_INFO("RAG disabled by config");
        return nullptr;
    }

    auto embed_cfg = get_embedding_config();
    auto embedder = std::make_shared<embedding::HttpEmbedding>(embed_cfg);

    auto store = std::make_shared<vectorstore::SqliteVectorStore>(embedder, mcfg.db_path);
    g_store = store;

    LOG_INFO("Vector store created: backend=sqlite path={}", mcfg.db_path);
    return store;
}

std::size_t ingest_documents(const std::string& dir)
{
    using namespace langchain;

    if (!g_store)
    {
        LOG_WARN("Cannot ingest: vector store not initialized");
        return 0;
    }

    if (!fs::exists(dir) || !fs::is_directory(dir))
    {
        LOG_WARN("Knowledge directory not found: {}", dir);
        return 0;
    }

    RagConfig rcfg = get_rag_config();

    std::vector<Document> all_docs;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (ext != ".txt" && ext != ".md")
            continue;

        try
        {
            document::TextFileLoader loader(entry.path().string());
            auto docs = loader.load();
            for (auto& d : docs)
            {
                d.metadata["source"] = entry.path().filename().string();
            }
            all_docs.insert(all_docs.end(), std::make_move_iterator(docs.begin()),
                            std::make_move_iterator(docs.end()));
        }
        catch (const std::exception& e)
        {
            LOG_WARN("Failed to load {}: {}", entry.path().string(), e.what());
        }
    }

    if (all_docs.empty())
    {
        LOG_INFO("No documents to ingest in: {}", dir);
        return 0;
    }

    text_splitter::RecursiveCharacterTextSplitter::Config split_cfg;
    split_cfg.chunk_size = rcfg.chunk_size;
    split_cfg.chunk_overlap = rcfg.chunk_overlap;
    text_splitter::RecursiveCharacterTextSplitter splitter(split_cfg);

    auto chunks = splitter.split_documents(std::move(all_docs));
    if (chunks.empty())
    {
        LOG_INFO("No chunks produced from documents");
        return 0;
    }

    try
    {
        g_store->add_documents(std::move(chunks));
        LOG_INFO("Ingested {} chunks from {}", chunks.size(), dir);
    }
    catch (const std::exception& e)
    {
        LOG_WARN("Failed to ingest documents into vector store: {}", e.what());
    }
    return chunks.size();
}

langchain::skill::RetrievalSkill build_retrieval_skill(
    langchain::llm::LLMPtr llm,
    std::shared_ptr<langchain::vectorstore::IVectorStore> store)
{
    using namespace langchain;

    RagConfig rcfg = get_rag_config();

    prompt::PromptTemplate tmpl(
        "Use the following retrieved context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:");

    return skill::RetrievalSkill(
        "rag_retriever",
        "Retrieve relevant documents and answer questions.",
        std::move(llm),
        std::move(store),
        std::move(tmpl),
        rcfg.retriever_top_k,
        "question",
        "context");
}

} // namespace smart_app
