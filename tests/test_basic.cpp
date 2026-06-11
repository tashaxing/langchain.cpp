// tests/test_basic.cpp — gtest-based smoke tests for core primitives.
#include <gtest/gtest.h>

#include "langchain.h"

#include <iostream>

using namespace langchain;

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

TEST(PromptTemplate, FormatsVariables)
{
    prompt::PromptTemplate t("Hello {who}, count={n}");
    EXPECT_EQ(t.input_variables().size(), 2u);

    auto out = t.format({{"who", "world"}, {"n", "3"}});
    EXPECT_EQ(out, "Hello world, count=3");
}

TEST(PromptTemplate, ThrowsOnMissingVariable)
{
    prompt::PromptTemplate t("Hello {who}, count={n}");
    EXPECT_THROW(t.format({{"who", "x"}}), LCError);
}

TEST(PromptTemplate, AllowMissingKeepsPlaceholder)
{
    prompt::PromptTemplate t("Hello {who}, count={n}");
    auto kept = t.format({{"who", "x"}}, /*allow_missing=*/true);
    EXPECT_EQ(kept, "Hello x, count={n}");
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

TEST(WindowMemory, EvictsOldExchanges)
{
    memory::WindowMemory w(/*k=*/2);
    w.add_exchange("u1", "a1");
    w.add_exchange("u2", "a2");
    w.add_exchange("u3", "a3"); // should evict u1/a1

    auto msgs = w.messages();
    EXPECT_EQ(msgs.size(), 4u);
    EXPECT_EQ(msgs.front().content, "u2");
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

TEST(CalculatorTool, EvaluatesExpression)
{
    auto t = tool::make_calculator_tool();
    auto out = t->invoke({{"expression", "(1+2)*3 - 4/2"}});
    EXPECT_EQ(out, "7");
}

// ---------------------------------------------------------------------------
// Embedding + VectorStore
// ---------------------------------------------------------------------------

TEST(HashingEmbeddingAndVectorStore, SimilaritySearch)
{
    auto emb = std::make_shared<embedding::HashingEmbedding>(64);
    vectorstore::InMemoryVectorStore vs(emb);
    vs.add_documents({
        {"", "the quick brown fox", {}, {}},
        {"", "lazy dog sleeps",     {}, {}},
        {"", "fox jumps over",      {}, {}}
    });

    auto hits = vs.similarity_search("fox", 2);
    EXPECT_FALSE(hits.empty());
    // The two fox docs should outrank the dog doc.
    EXPECT_NE(hits[0].doc.content.find("fox"), std::string::npos);
}
