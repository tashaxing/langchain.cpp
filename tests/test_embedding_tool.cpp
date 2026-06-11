// tests/test_embedding_tool.cpp — Embedding and Tool unit tests.
#include <gtest/gtest.h>

#include "embedding/embedding.h"
#include "tool/tool.h"

using namespace langchain;

// ---------------------------------------------------------------------------
// HashingEmbedding
// ---------------------------------------------------------------------------
TEST(HashingEmbedding, DimensionMatchesConfig)
{
    embedding::HashingEmbedding emb(64);
    EXPECT_EQ(emb.dimension(), 64);
}

TEST(HashingEmbedding, SameTextSameVector)
{
    embedding::HashingEmbedding emb(64);
    auto a = emb.embed_query("hello");
    auto b = emb.embed_query("hello");

    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}

TEST(HashingEmbedding, DifferentTextDifferentVector)
{
    embedding::HashingEmbedding emb(64);
    auto a = emb.embed_query("hello");
    auto b = emb.embed_query("world");

    bool all_same = true;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        if (a[i] != b[i]) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same);
}

TEST(HashingEmbedding, BatchEmbed)
{
    embedding::HashingEmbedding emb(32);
    auto vecs = emb.embed_documents({"a", "b", "c"});

    ASSERT_EQ(vecs.size(), 3u);
    for (const auto& v : vecs)
    {
        EXPECT_EQ(v.size(), 32u);
    }
}

TEST(HashingEmbedding, EmptyText)
{
    embedding::HashingEmbedding emb(16);
    auto vec = emb.embed_query("");
    EXPECT_EQ(vec.size(), 16u);
}

TEST(HashingEmbedding, L2Normalized)
{
    embedding::HashingEmbedding emb(8);
    auto vec = emb.embed_query("some text here");

    double norm = 0.0;
    for (float v : vec) { norm += v * v; }
    norm = std::sqrt(norm);
    EXPECT_FLOAT_EQ(static_cast<float>(norm), 1.0f);
}

// ---------------------------------------------------------------------------
// HttpEmbedding / OllamaEmbedding / LocalAIEmbedding — config only
// ---------------------------------------------------------------------------
TEST(EmbeddingConfig, HttpEmbeddingDefaults)
{
    embedding::HttpEmbeddingConfig cfg;
    EXPECT_EQ(cfg.base_url, "https://api.openai.com");
    EXPECT_EQ(cfg.model, "text-embedding-3-small");
    EXPECT_EQ(cfg.dimension, 1536);
}

TEST(EmbeddingConfig, OllamaEmbeddingDefaults)
{
    embedding::OllamaEmbeddingConfig cfg;
    EXPECT_EQ(cfg.base_url, "http://localhost:11434");
    EXPECT_EQ(cfg.model, "nomic-embed-text");
}

TEST(EmbeddingConfig, LocalAIEmbeddingDefaults)
{
    embedding::LocalAIEmbeddingConfig cfg;
    EXPECT_EQ(cfg.base_url, "http://localhost:8080");
}

// ---------------------------------------------------------------------------
// Calculator tool
// ---------------------------------------------------------------------------
TEST(CalculatorTool, BasicArithmetic)
{
    auto t = tool::make_calculator_tool();
    auto out = t->invoke({{"expression", "(1+2)*3 - 4/2"}});
    EXPECT_EQ(out, "7");
}

TEST(CalculatorTool, Division)
{
    auto t = tool::make_calculator_tool();
    auto out = t->invoke({{"expression", "10 / 4"}});
    EXPECT_EQ(out, "2.5");
}

TEST(CalculatorTool, MissingExpression)
{
    auto t = tool::make_calculator_tool();
    auto out = t->invoke(json::object());
    EXPECT_NE(out.find("missing"), std::string::npos);
}

TEST(CalculatorTool, InvalidExpression)
{
    auto t = tool::make_calculator_tool();
    // "1 + * 2" is genuinely invalid: '*' is not a valid unary operator.
    auto out = t->invoke({{"expression", "1 + * 2"}});
    EXPECT_NE(out.find("error"), std::string::npos);
}


// ---------------------------------------------------------------------------
// HttpGet tool — config only (no network in unit tests)
// ---------------------------------------------------------------------------
TEST(HttpGetTool, SchemaHasUrl)
{
    auto t = tool::make_http_get_tool();
    auto schema = t->parameters_schema();
    EXPECT_TRUE(schema.contains("properties"));
    EXPECT_TRUE(schema["properties"].contains("url"));
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------
TEST(ToolRegistry, AddGet)
{
    tool::ToolRegistry reg;
    reg.add(tool::make_calculator_tool());

    EXPECT_EQ(reg.size(), 1u);
    EXPECT_NE(reg.get("calculator"), nullptr);
    EXPECT_EQ(reg.get("nope"), nullptr);
}

TEST(ToolRegistry, GetOrThrow)
{
    tool::ToolRegistry reg;
    reg.add(tool::make_calculator_tool());

    EXPECT_NO_THROW(reg.get_or_throw("calculator"));
    EXPECT_THROW(reg.get_or_throw("nope"), LCError);
}

TEST(ToolRegistry, NamesAndSchemas)
{
    tool::ToolRegistry reg;
    reg.add(tool::make_calculator_tool());
    reg.add(tool::make_http_get_tool());

    auto names = reg.names();
    EXPECT_EQ(names.size(), 2u);

    auto schemas = reg.schemas();
    EXPECT_EQ(schemas.size(), 2u);
}

TEST(ToolRegistry, Merge)
{
    tool::ToolRegistry a;
    a.add(tool::make_calculator_tool());

    tool::ToolRegistry b;
    b.add(tool::make_http_get_tool());

    a.merge(b);
    EXPECT_EQ(a.size(), 2u);
}

TEST(ToolRegistry, Empty)
{
    tool::ToolRegistry reg;
    EXPECT_TRUE(reg.empty());
    reg.add(tool::make_calculator_tool());
    EXPECT_FALSE(reg.empty());
}

// ---------------------------------------------------------------------------
// FunctionTool
// ---------------------------------------------------------------------------
TEST(FunctionTool, LambdaInvocation)
{
    auto t = std::make_shared<tool::FunctionTool>(
        "double", "doubles a number",
        json{{"type", "object"},
             {"properties", {{"n", {{"type", "number"}}}}},
             {"required", json::array({"n"})}},
        [](const json& args) -> std::string
        {
            int n = args.value("n", 0);
            return std::to_string(n * 2);
        });

    auto out = t->invoke({{"n", 21}});
    EXPECT_EQ(out, "42");
}

// ---------------------------------------------------------------------------
// JSON Schema validation
// ---------------------------------------------------------------------------
TEST(ToolValidation, RequiredFieldMissing)
{
    auto t = tool::make_calculator_tool();
    auto result = t->validate(json::object());
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.error.find("missing required"), std::string::npos);
}

TEST(ToolValidation, WrongType)
{
    auto t = tool::make_calculator_tool();
    auto result = t->validate({{"expression", 42}});
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.error.find("expected type 'string'"), std::string::npos);
}

TEST(ToolValidation, ValidArgs)
{
    auto t = tool::make_calculator_tool();
    auto result = t->validate({{"expression", "1+2"}});
    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.error.empty());
}

TEST(ToolValidation, NumberType)
{
    auto t = std::make_shared<tool::FunctionTool>(
        "test", "test",
        json{{"type", "object"},
             {"properties", {{"count", {{"type", "integer"}}}}},
             {"required", json::array({"count"})}},
        [](const json&) -> std::string { return "ok"; });

    EXPECT_TRUE(t->validate({{"count", 5}}).valid);
    EXPECT_FALSE(t->validate({{"count", "five"}}).valid);
}

TEST(ToolValidation, BooleanType)
{
    auto t = std::make_shared<tool::FunctionTool>(
        "test", "test",
        json{{"type", "object"},
             {"properties", {{"flag", {{"type", "boolean"}}}}},
             {"required", json::array({"flag"})}},
        [](const json&) -> std::string { return "ok"; });

    EXPECT_TRUE(t->validate({{"flag", true}}).valid);
    EXPECT_FALSE(t->validate({{"flag", "true"}}).valid);
}

TEST(ToolValidation, ArrayType)
{
    auto t = std::make_shared<tool::FunctionTool>(
        "test", "test",
        json{{"type", "object"},
             {"properties", {{"items", {{"type", "array"}}}}},
             {"required", json::array({"items"})}},
        [](const json&) -> std::string { return "ok"; });

    EXPECT_TRUE(t->validate({{"items", json::array({1, 2, 3})}}).valid);
    EXPECT_FALSE(t->validate({{"items", "not array"}}).valid);
}

TEST(ToolValidation, OptionalFieldMissingIsOk)
{
    auto t = std::make_shared<tool::FunctionTool>(
        "test", "test",
        json{{"type", "object"},
             {"properties", {{"required_field", {{"type", "string"}}},
                             {"optional_field", {{"type", "string"}}}}},
             {"required", json::array({"required_field"})}},
        [](const json&) -> std::string { return "ok"; });

    EXPECT_TRUE(t->validate({{"required_field", "x"}}).valid);
}
