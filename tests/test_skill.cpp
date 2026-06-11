// tests/test_skill.cpp — Skill unit tests.
#include <gtest/gtest.h>

#include "skill/skill.h"
#include "skill/skill_registry.h"
#include "llm/llm.h"

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// MockLLM — records prompts and returns canned responses.
// ---------------------------------------------------------------------------
class MockLLM : public llm::ILLM
{
public:
    std::vector<std::string> prompts;
    std::string canned = "canned-response";

    std::string name() const override { return "mock"; }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
    {
        std::string last;
        for (const auto& m : req.messages)
        {
            if (m.role == Role::User) { last = m.content; }
        }
        prompts.push_back(last);
        llm::ChatResponse out;
        out.message = Message::assistant(canned);
        out.finish_reason = "stop";
        return out;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// PromptSkill
// ---------------------------------------------------------------------------
TEST(PromptSkill, FormatsAndCallsLLM)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl("Hello {name}");
    skill::PromptSkill s("greet", "Greets someone", llm, tmpl);

    skill::SkillContext ctx;
    ctx.vars["name"] = "world";
    auto out = s.invoke(ctx);

    EXPECT_EQ(out, "canned-response");
    ASSERT_EQ(llm->prompts.size(), 1u);
    EXPECT_EQ(llm->prompts[0], "Hello world");
}

// ---------------------------------------------------------------------------
// RetrievalSkill
// ---------------------------------------------------------------------------
TEST(RetrievalSkill, InjectsContextIntoPrompt)
{
    auto llm = std::make_shared<MockLLM>();

    // Fake embedder that returns fixed-size zero vectors.
    class FakeEmbedder : public embedding::IEmbedding
    {
    public:
        std::vector<std::vector<float>> embed_documents(
            const std::vector<std::string>& texts) override
        {
            return std::vector<std::vector<float>>(texts.size(),
                                                   std::vector<float>(4, 0.0f));
        }
        int dimension() const override { return 4; }
        std::string name() const override { return "fake"; }
    };

    auto embedder = std::make_shared<FakeEmbedder>();
    auto vs = std::make_shared<vectorstore::InMemoryVectorStore>(embedder);
    vs->add_documents({{"", "doc1", {}, {}}, {"", "doc2", {}, {}}});

    prompt::PromptTemplate tmpl("Q: {question}\nContext: {context}");
    skill::RetrievalSkill s("rag", "RAG skill", llm,
                            vs,
                            tmpl, 2);

    skill::SkillContext ctx;
    ctx.vars["question"] = "what?";
    auto out = s.invoke(ctx);

    EXPECT_EQ(out, "canned-response");
    ASSERT_EQ(llm->prompts.size(), 1u);
    EXPECT_NE(llm->prompts[0].find("doc1"), std::string::npos);
}

// ---------------------------------------------------------------------------
// ChainSkill
// ---------------------------------------------------------------------------
TEST(ChainSkill, SequentialExecution)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl1("Step1: {input}");
    prompt::PromptTemplate tmpl2("Step2: {step1_out}");

    skill::ChainSkill chain("pipeline", "two-step",
        {
            {std::make_shared<skill::PromptSkill>("s1", "", llm, tmpl1), "step1_out"},
            {std::make_shared<skill::PromptSkill>("s2", "", llm, tmpl2), "step2_out"},
        });

    skill::SkillContext ctx;
    ctx.vars["input"] = "hello";
    auto out = chain.invoke(ctx);

    EXPECT_EQ(out, "canned-response");
    ASSERT_EQ(llm->prompts.size(), 2u);
    EXPECT_EQ(llm->prompts[0], "Step1: hello");
    EXPECT_EQ(llm->prompts[1], "Step2: canned-response");
}

// ---------------------------------------------------------------------------
// ChainSkill — run_traced returns intermediates
// ---------------------------------------------------------------------------
TEST(ChainSkill, RunTracedReturnsIntermediates)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl("T: {input}");

    skill::ChainSkill chain("traced", "",
        {
            {std::make_shared<skill::PromptSkill>("s1", "", llm, tmpl), "out1"},
        });

    skill::SkillContext ctx;
    ctx.vars["input"] = "x";
    auto trace = chain.run_traced(ctx);

    EXPECT_EQ(trace["out1"], "canned-response");
}

// ---------------------------------------------------------------------------
// RouterSkill
// ---------------------------------------------------------------------------
TEST(RouterSkill, RoutesByKey)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl_a("A");
    prompt::PromptTemplate tmpl_b("B");

    skill::RouterSkill router("router", "",
        "branch",
        {
            {"a", std::make_shared<skill::PromptSkill>("sa", "", llm, tmpl_a)},
            {"b", std::make_shared<skill::PromptSkill>("sb", "", llm, tmpl_b)},
        });

    skill::SkillContext ctx;
    ctx.vars["branch"] = "b";
    auto out = router.invoke(ctx);

    ASSERT_EQ(llm->prompts.size(), 1u);
    EXPECT_EQ(llm->prompts[0], "B");
}

TEST(RouterSkill, MissingKeyThrows)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl("X");

    skill::RouterSkill router("router", "", "key", {});

    skill::SkillContext ctx;
    EXPECT_THROW(router.invoke(ctx), LCError);
}

// ---------------------------------------------------------------------------
// SkillRegistry
// ---------------------------------------------------------------------------
TEST(SkillRegistry, AddGetNames)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl("T");

    skill::SkillRegistry reg;
    reg.add(std::make_shared<skill::PromptSkill>("alpha", "desc", llm, tmpl));
    reg.add(std::make_shared<skill::PromptSkill>("beta",  "desc", llm, tmpl));

    EXPECT_EQ(reg.size(), 2u);
    auto names = reg.names();
    EXPECT_TRUE(std::find(names.begin(), names.end(), "alpha") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "beta")  != names.end());
}

TEST(SkillRegistry, AsToolsExportsSkills)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl("T");

    skill::SkillRegistry reg;
    reg.add(std::make_shared<skill::PromptSkill>("skill1", "does thing", llm, tmpl));

    auto tools = reg.as_tools();
    EXPECT_EQ(tools.size(), 1u);
    EXPECT_TRUE(tools.get("skill1") != nullptr);
}

// ---------------------------------------------------------------------------
// skill_to_tool
// ---------------------------------------------------------------------------
TEST(SkillToTool, ConvertsSkillToTool)
{
    auto llm = std::make_shared<MockLLM>();
    prompt::PromptTemplate tmpl("Say {word}");
    auto skill_ptr = std::make_shared<skill::PromptSkill>("talk", "talks", llm, tmpl);

    auto t = skill::skill_to_tool(skill_ptr);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->name(), "talk");

    auto result = t->invoke(json{{"word", "hi"}});
    EXPECT_EQ(result, "canned-response");
}
