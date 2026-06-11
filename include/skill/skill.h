// langchain/skill/skill.h
// A Skill is a packaged capability — prompt + (optional) tools + (optional)
// retriever — that an agent or app can invoke as one unit. Analogous to a
// LangChain "Chain" + "Skill" hybrid; intentionally small.
#pragma once

#include "llm/llm.h"
#include "prompt/prompt_template.h"
#include "retriever/retriever.h"
#include "tool/tool.h"
#include "vectorstore/vectorstore.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace langchain
{
namespace hook
{
class HookManager;
}
}

namespace langchain
{
namespace skill
{

struct SkillContext
{
    std::unordered_map<std::string, std::string> vars;
};

class ISkill
{
public:
    virtual ~ISkill() = default;
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual std::string invoke(const SkillContext& ctx) = 0;

    void set_hooks(hook::HookManager* mgr)
    {
        hooks_ = mgr;
    }
    hook::HookManager* hooks() const
    {
        return hooks_;
    }

private:
    hook::HookManager* hooks_ = nullptr;
};

using SkillPtr = std::shared_ptr<ISkill>;

// Single-shot LLM skill: render a prompt template then call the LLM once.
class PromptSkill : public ISkill
{
public:
    PromptSkill(std::string name,
                std::string desc,
                llm::LLMPtr llm,
                prompt::PromptTemplate tmpl);

    std::string name() const override;
    std::string description() const override;
    std::string invoke(const SkillContext& ctx) override;

private:
    std::string name_;
    std::string desc_;
    llm::LLMPtr llm_;
    prompt::PromptTemplate tmpl_;
};

// Retrieval-augmented skill: pull top-k docs from a vector store, stuff them
// into the prompt under {context}, then call the LLM.
// Configurable keys let you adapt the skill to different prompt templates.
class RetrievalSkill : public ISkill
{
public:
    // result_formatter: function that formats retrieved documents into a string.
    //                  If null, a default formatter is used.
    RetrievalSkill(std::string name,
                   std::string desc,
                   llm::LLMPtr llm,
                   vectorstore::VectorStorePtr vs,
                   prompt::PromptTemplate tmpl,
                   int k = 4,
                   std::string query_key = "question",
                   std::string context_key = "context");

    std::string name() const override;
    std::string description() const override;
    std::string invoke(const SkillContext& ctx) override;

private:
    std::string name_;
    std::string desc_;
    llm::LLMPtr llm_;
    vectorstore::VectorStorePtr vs_;
    prompt::PromptTemplate tmpl_;
    int k_;
    std::string query_key_;
    std::string context_key_;

    static std::string default_format_(const std::vector<vectorstore::ScoredDocument>& docs);
};

// Chain multiple skills sequentially. Each step's output is bound into the
// shared context under `output_key`, so later steps' prompt templates can
// reference {output_key}. Resembles LangChain's SequentialChain.
class ChainSkill : public ISkill
{
public:
    struct Step
    {
        SkillPtr skill;
        std::string output_key; // e.g. "summary" => later steps see {summary}
    };

    ChainSkill(std::string name, std::string desc, std::vector<Step> steps);

    std::string name() const override;
    std::string description() const override;
    std::string invoke(const SkillContext& ctx) override;

    // Run and return every intermediate output keyed by the step's output_key.
    std::unordered_map<std::string, std::string> run_traced(const SkillContext& ctx);

private:
    std::string name_;
    std::string desc_;
    std::vector<Step> steps_;
};

// Branch on a key in the context: pick whichever case matches ctx.vars[key].
// Useful for "router" patterns analogous to LangChain's MultiPromptChain.
class RouterSkill : public ISkill
{
public:
    RouterSkill(std::string name,
                std::string desc,
                std::string key,
                std::unordered_map<std::string, SkillPtr> cases,
                SkillPtr default_skill = nullptr);

    std::string name() const override;
    std::string description() const override;
    std::string invoke(const SkillContext& ctx) override;

private:
    std::string name_;
    std::string desc_;
    std::string key_;
    std::unordered_map<std::string, SkillPtr> cases_;
    SkillPtr default_;
};

// ---------------- ScriptSkill ----------------
// Executes an external script (shell .sh or batch .bat) and captures stdout.
// Parameters from SkillContext::vars are passed as positional arguments.
class ScriptSkill : public ISkill
{
public:
    ScriptSkill(std::string name,
                std::string desc,
                std::string script_path);

    std::string name() const override;
    std::string description() const override;
    std::string invoke(const SkillContext& ctx) override;

private:
    std::string name_;
    std::string desc_;
    std::string script_path_;

    static std::string exec_(const std::string& cmd);
};

// Convert a skill into a tool that an agent can invoke.
// The tool schema is derived from the skill's name and description.
// Arguments are passed as JSON and converted to SkillContext::vars.
tool::ToolPtr skill_to_tool(SkillPtr skill);

} // namespace skill
} // namespace langchain
