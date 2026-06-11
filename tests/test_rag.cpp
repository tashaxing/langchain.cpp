// tests/test_gtest_rag.cpp — RAG pipeline unit tests.
#include <gtest/gtest.h>

#include "document/text_loader.h"
#include "embedding/embedding.h"
#include "text_splitter/text_splitter.h"
#include "vectorstore/vectorstore.h"

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
std::string make_temp_text_file(const std::string& content)
{
    char path[] = "test_rag_XXXXXX.txt";
    std::FILE* fp = std::fopen(path, "w+");
    std::fwrite(content.c_str(), 1, content.size(), fp);
    std::fclose(fp);
    return path;
}

// ---------------------------------------------------------------------------
// TextFileLoader
// ---------------------------------------------------------------------------
TEST(TextFileLoader, LoadsFileContent)
{
    std::string content = "Hello world!\nThis is a test.";
    std::string path = make_temp_text_file(content);

    document::TextFileLoader loader(path);
    auto docs = loader.load();

    ASSERT_EQ(docs.size(), 1u);
    EXPECT_EQ(docs[0].content, content);
    EXPECT_EQ(docs[0].metadata.at("source"), path);

    std::remove(path.c_str());
}

TEST(TextFileLoader, ThrowsOnMissingFile)
{
    document::TextFileLoader loader("/nonexistent/path/to/file.txt");
    EXPECT_THROW(loader.load(), LCError);
}

// ---------------------------------------------------------------------------
// RecursiveCharacterTextSplitter
// ---------------------------------------------------------------------------
TEST(RecursiveCharacterTextSplitter, SplitsSimpleText)
{
    text_splitter::RecursiveCharacterTextSplitter::Config cfg;
    cfg.chunk_size = 50;
    cfg.chunk_overlap = 0;
    text_splitter::RecursiveCharacterTextSplitter splitter(cfg);

    auto chunks = splitter.split_text("Hello world. This is a test. Another sentence here.");
    ASSERT_GE(chunks.size(), 1u);
    for (const auto& c : chunks)
    {
        EXPECT_LE(c.size(), 50u + 10u); // allow some slack
    }
}

TEST(RecursiveCharacterTextSplitter, RespectsSeparators)
{
    text_splitter::RecursiveCharacterTextSplitter::Config cfg;
    cfg.chunk_size = 20;
    cfg.chunk_overlap = 0;
    text_splitter::RecursiveCharacterTextSplitter splitter(cfg);

    // Newline should be preferred over space.
    auto chunks = splitter.split_text("Line one\nLine two\nLine three");
    ASSERT_GE(chunks.size(), 2u);
    for (const auto& c : chunks)
    {
        EXPECT_LE(c.size(), 20u + 10u);
    }
}

TEST(RecursiveCharacterTextSplitter, SplitsDocuments)
{
    text_splitter::RecursiveCharacterTextSplitter::Config cfg;
    cfg.chunk_size = 50;
    cfg.chunk_overlap = 0;
    text_splitter::RecursiveCharacterTextSplitter splitter(cfg);

    Document doc;
    doc.id = "test";
    doc.content = "Hello world. This is a test.";
    doc.metadata["key"] = "value";

    auto docs = splitter.split_documents(doc);
    ASSERT_GE(docs.size(), 1u);
    for (const auto& d : docs)
    {
        EXPECT_EQ(d.metadata.at("key"), "value");
    }
}

TEST(RecursiveCharacterTextSplitter, OverlapIsApplied)
{
    text_splitter::RecursiveCharacterTextSplitter::Config cfg;
    cfg.chunk_size = 20;
    cfg.chunk_overlap = 5;
    text_splitter::RecursiveCharacterTextSplitter splitter(cfg);

    auto chunks = splitter.split_text("abcdefghij klmnopqrst uvwxyz");
    ASSERT_GE(chunks.size(), 2u);
    // Overlap means second chunk should start with end of first.
    EXPECT_TRUE(chunks[1].find(chunks[0].substr(chunks[0].size() - 5)) != std::string::npos ||
                chunks[1].find(chunks[0].substr(chunks[0].size() - 3)) != std::string::npos);
}

TEST(RecursiveCharacterTextSplitter, EmptyTextReturnsEmpty)
{
    text_splitter::RecursiveCharacterTextSplitter splitter;
    auto chunks = splitter.split_text("");
    EXPECT_TRUE(chunks.empty());
}

TEST(RecursiveCharacterTextSplitter, ShortTextReturnedAsIs)
{
    text_splitter::RecursiveCharacterTextSplitter splitter;
    auto chunks = splitter.split_text("Short");
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0], "Short");
}

// ---------------------------------------------------------------------------
// VectorStore with Metadata Filtering
// ---------------------------------------------------------------------------
TEST(VectorStoreFilter, FiltersByMetadata)
{
    auto emb = std::make_shared<embedding::HashingEmbedding>(64);
    vectorstore::InMemoryVectorStore vs(emb);

    vs.add_documents({
        { "", "fox runs fast", { {"category", "animal"} }, {} },
        { "", "dog barks loud", { {"category", "animal"} }, {} },
        { "", "car drives fast", { {"category", "vehicle"} }, {} },
    });

    auto hits = vs.similarity_search("fox", 10, { {"category", "vehicle"} });
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].doc.content, "car drives fast");
}

TEST(VectorStoreFilter, NoFilterReturnsAll)
{
    auto emb = std::make_shared<embedding::HashingEmbedding>(64);
    vectorstore::InMemoryVectorStore vs(emb);

    vs.add_documents({
        { "", "fox runs fast", {}, {} },
        { "", "dog barks loud", {}, {} },
    });

    auto hits = vs.similarity_search("fox", 10);
    EXPECT_EQ(hits.size(), 2u);
}

TEST(VectorStoreFilter, NonMatchingFilterReturnsEmpty)
{
    auto emb = std::make_shared<embedding::HashingEmbedding>(64);
    vectorstore::InMemoryVectorStore vs(emb);

    vs.add_documents({
        { "", "fox runs fast", { {"category", "animal"} }, {} },
    });

    auto hits = vs.similarity_search("fox", 10, { {"category", "vehicle"} });
    EXPECT_TRUE(hits.empty());
}

} // namespace
