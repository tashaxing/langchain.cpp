// tests/test_vectorstore.cpp — VectorStore unit tests.
//
// Covers InMemoryVectorStore, SqliteVectorStore, and CRUD operations.
#include <gtest/gtest.h>

#include "vectorstore/vectorstore.h"
#include "embedding/embedding.h"

#include <chrono>
#include <cstdio>
#include <filesystem>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// FakeEmbedder — deterministic small vectors for testing.
// ---------------------------------------------------------------------------
class FakeEmbedder : public embedding::IEmbedding
{
public:
    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override
    {
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (std::size_t i = 0; i < texts.size(); ++i)
        {
            // Deterministic: first element encodes index, rest are zero.
            std::vector<float> v(4, 0.0f);
            v[0] = static_cast<float>(i + 1);
            out.push_back(std::move(v));
        }
        return out;
    }

    int dimension() const override { return 4; }
    std::string name() const override { return "fake"; }
};

std::string make_temp_db()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = std::filesystem::temp_directory_path() /
                ("lc_vs_test_" + std::to_string(now) + ".db");
    return path.string();
}

} // namespace

// ---------------------------------------------------------------------------
// InMemoryVectorStore — CRUD
// ---------------------------------------------------------------------------
TEST(InMemoryVectorStore, AddAndSearch)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);

    vs.add_documents({{"", "fox runs", {}, {}}, {"", "dog barks", {}, {}}});
    EXPECT_EQ(vs.size(), 2u);

    auto hits = vs.similarity_search("fox", 2);
    ASSERT_EQ(hits.size(), 2u);
    EXPECT_EQ(hits[0].doc.content, "fox runs"); // highest score
}

TEST(InMemoryVectorStore, DeleteDocuments)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);

    auto ids = vs.add_documents({{"id-a", "aaa", {}, {}}, {"id-b", "bbb", {}, {}}});
    ASSERT_EQ(vs.size(), 2u);

    std::size_t removed = vs.delete_documents({"id-a"});
    EXPECT_EQ(removed, 1u);
    EXPECT_EQ(vs.size(), 1u);

    // After deleting id-a, only id-b remains.  Searching for "aaa" (which
    // embeds to the same vector as the deleted doc) still returns id-b because
    // FakeEmbedder gives every document a non-zero cosine with every query.
    // Instead verify that the *remaining* doc is id-b.
    auto hits = vs.similarity_search("anything", 10);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].doc.id, "id-b");
}

TEST(InMemoryVectorStore, UpdateDocument)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);

    vs.add_documents({{"id-x", "old content", {}, {}}});

    Document updated;
    updated.content = "new content";
    vs.update_document("id-x", std::move(updated));

    auto hits = vs.similarity_search("new content", 10);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].doc.content, "new content");
}

TEST(InMemoryVectorStore, UpdateMissingThrows)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);
    Document doc;
    doc.content = "x";
    EXPECT_THROW(vs.update_document("nope", std::move(doc)), LCError);
}

TEST(InMemoryVectorStore, FilterByMetadata)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);

    vs.add_documents({
        {"", "fox",  {{"type", "animal"}}, {}},
        {"", "car", {{"type", "vehicle"}}, {}},
    });

    auto hits = vs.similarity_search("anything", 10, {{"type", "vehicle"}});
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].doc.content, "car");
}

// ---------------------------------------------------------------------------
// SqliteVectorStore — persistence
// ---------------------------------------------------------------------------
TEST(SqliteVectorStore, PersistenceRoundTrip)
{
    std::string path = make_temp_db();
    std::vector<std::string> ids;
    {
        auto emb = std::make_shared<FakeEmbedder>();
        vectorstore::SqliteVectorStore vs(emb, path);
        ids = vs.add_documents({{"", "persist me", {{"k","v"}}, {}}});
        EXPECT_EQ(vs.size(), 1u);
    }
    {
        auto emb = std::make_shared<FakeEmbedder>();
        vectorstore::SqliteVectorStore vs(emb, path);
        EXPECT_EQ(vs.size(), 1u);

        auto hits = vs.similarity_search("persist", 10);
        ASSERT_EQ(hits.size(), 1u);
        EXPECT_EQ(hits[0].doc.content, "persist me");
        EXPECT_EQ(hits[0].doc.metadata.at("k"), "v");
    }
    std::remove(path.c_str());
}

TEST(SqliteVectorStore, DeletePersists)
{
    std::string path = make_temp_db();
    {
        auto emb = std::make_shared<FakeEmbedder>();
        vectorstore::SqliteVectorStore vs(emb, path);
        vs.add_documents({{"id-1", "a", {}, {}}, {"id-2", "b", {}, {}}});
        vs.delete_documents({"id-1"});
        EXPECT_EQ(vs.size(), 1u);
    }
    {
        auto emb = std::make_shared<FakeEmbedder>();
        vectorstore::SqliteVectorStore vs(emb, path);
        EXPECT_EQ(vs.size(), 1u);
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------
TEST(VectorStore, EmptySearchReturnsEmpty)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);
    EXPECT_TRUE(vs.similarity_search("anything", 5).empty());
}

TEST(VectorStore, ZeroKReturnsEmpty)
{
    auto emb = std::make_shared<FakeEmbedder>();
    vectorstore::InMemoryVectorStore vs(emb);
    vs.add_documents({{"", "x", {}, {}}});
    EXPECT_TRUE(vs.similarity_search("x", 0).empty());
}
