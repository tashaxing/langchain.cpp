// src/skill/skill.cpp — implementations for the skill family.
#include "skill/skill.h"

#include "hook/hook.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <utility>

namespace langchain
{
namespace skill
{

namespace
{

// Render SkillContext::vars to a compact "k=v; k=v" string for the hook event.
std::string serialize_ctx(const SkillContext& ctx)
{
    std::ostringstream oss;
    bool first = true;
    for (const auto& kv : ctx.vars)
    {
        if (!first)
        {
            oss << "; ";
        }
        first = false;
        oss << kv.first << "=" << kv.second;
    }
    return oss.str();
}

hook::ScopedSpan begin_skill_span(ISkill& self, const SkillContext& ctx,
                                  std::string* input_buf)
{
    auto* mgr = self.hooks() ? self.hooks() : &hook::HookManager::global();
    *input_buf = serialize_ctx(ctx);

    hook::HookContext before;
    before.phase        = hook::Phase::BeforeSkill;
    before.component    = self.name();
    before.call_id      = mgr->new_call_id();
    before.skill_input  = input_buf;
    return hook::ScopedSpan(mgr, before, hook::Phase::AfterSkill);
}

} // namespace

// ---------------- PromptSkill ----------------

PromptSkill::PromptSkill(std::string name,
                         std::string desc,
                         llm::LLMPtr llm,
                         prompt::PromptTemplate tmpl)
    : name_(std::move(name)),
      desc_(std::move(desc)),
      llm_(std::move(llm)),
      tmpl_(std::move(tmpl))
{
}

std::string PromptSkill::name() const
{
    return name_;
}

std::string PromptSkill::description() const
{
    return desc_;
}

std::string PromptSkill::invoke(const SkillContext& ctx)
{
    std::string buf;
    auto span = begin_skill_span(*this, ctx, &buf);
    std::string p = tmpl_.format(ctx.vars);
    std::string out = llm_->complete(p);
    span.after().skill_output = &out;
    return out;
}

// ---------------- RetrievalSkill ----------------

RetrievalSkill::RetrievalSkill(std::string name,
                               std::string desc,
                               llm::LLMPtr llm,
                               vectorstore::VectorStorePtr vs,
                               prompt::PromptTemplate tmpl,
                               int k,
                               std::string query_key,
                               std::string context_key)
    : name_(std::move(name)),
      desc_(std::move(desc)),
      llm_(std::move(llm)),
      vs_(std::move(vs)),
      tmpl_(std::move(tmpl)),
      k_(k),
      query_key_(std::move(query_key)),
      context_key_(std::move(context_key))
{
}

std::string RetrievalSkill::name() const
{
    return name_;
}

std::string RetrievalSkill::description() const
{
    return desc_;
}

std::string RetrievalSkill::default_format_(const std::vector<vectorstore::ScoredDocument>& docs)
{
    std::string out;
    for (const auto& h : docs)
    {
        out += "- ";
        out += h.doc.content;
        out += "\n";
    }
    return out;
}

std::string RetrievalSkill::invoke(const SkillContext& ctx)
{
    std::string buf;
    auto span = begin_skill_span(*this, ctx, &buf);

    auto qit = ctx.vars.find(query_key_);
    std::string q = (qit == ctx.vars.end()) ? std::string() : qit->second;
    auto hits = vs_->similarity_search(q, k_);
    std::string ctx_text = default_format_(hits);
    auto vars = ctx.vars;
    vars[context_key_] = ctx_text;
    std::string out = llm_->complete(tmpl_.format(vars));
    span.after().skill_output = &out;
    return out;
}

// ---------------- ChainSkill ----------------

ChainSkill::ChainSkill(std::string name, std::string desc, std::vector<Step> steps)
    : name_(std::move(name)),
      desc_(std::move(desc)),
      steps_(std::move(steps))
{
}

std::string ChainSkill::name() const
{
    return name_;
}

std::string ChainSkill::description() const
{
    return desc_;
}

std::string ChainSkill::invoke(const SkillContext& ctx)
{
    std::string buf;
    auto span = begin_skill_span(*this, ctx, &buf);

    SkillContext cur = ctx;
    std::string last;
    for (auto& s : steps_)
    {
        last = s.skill->invoke(cur);
        if (!s.output_key.empty())
        {
            cur.vars[s.output_key] = last;
        }
    }
    span.after().skill_output = &last;
    return last;
}

std::unordered_map<std::string, std::string>
ChainSkill::run_traced(const SkillContext& ctx)
{
    SkillContext cur = ctx;
    std::unordered_map<std::string, std::string> trace;
    for (auto& s : steps_)
    {
        std::string out = s.skill->invoke(cur);
        std::string key = s.output_key.empty() ? s.skill->name() : s.output_key;
        trace[key] = out;
        if (!s.output_key.empty())
        {
            cur.vars[s.output_key] = out;
        }
    }
    return trace;
}

// ---------------- RouterSkill ----------------

RouterSkill::RouterSkill(std::string name,
                         std::string desc,
                         std::string key,
                         std::unordered_map<std::string, SkillPtr> cases,
                         SkillPtr default_skill)
    : name_(std::move(name)),
      desc_(std::move(desc)),
      key_(std::move(key)),
      cases_(std::move(cases)),
      default_(std::move(default_skill))
{
}

std::string RouterSkill::name() const
{
    return name_;
}

std::string RouterSkill::description() const
{
    return desc_;
}

std::string RouterSkill::invoke(const SkillContext& ctx)
{
    auto it = ctx.vars.find(key_);
    if (it != ctx.vars.end())
    {
        auto found = cases_.find(it->second);
        if (found != cases_.end())
        {
            return found->second->invoke(ctx);
        }
    }
    if (default_)
    {
        return default_->invoke(ctx);
    }
    throw LCError("RouterSkill: no branch matched key '" + key_ + "'");
}

// ---------------- ScriptSkill ----------------

ScriptSkill::ScriptSkill(std::string name,
                         std::string desc,
                         std::string script_path)
    : name_(std::move(name)),
      desc_(std::move(desc)),
      script_path_(std::move(script_path))
{
}

std::string ScriptSkill::name() const
{
    return name_;
}

std::string ScriptSkill::description() const
{
    return desc_;
}

std::string ScriptSkill::invoke(const SkillContext& ctx)
{
    std::string buf;
    auto span = begin_skill_span(*this, ctx, &buf);

#ifdef _WIN32
    std::string cmd = "cmd /c \"\"\"" + script_path_ + ".bat\"\"\"";
#else
    std::string cmd = "bash \"" + script_path_ + ".sh\"";
#endif

    // Append arguments from ctx.vars in alphabetical order for determinism.
    std::vector<std::string> keys;
    keys.reserve(ctx.vars.size());
    for (const auto& kv : ctx.vars)
    {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys)
    {
        cmd += " \"" + ctx.vars.at(k) + "\"";
    }

    std::string out = exec_(cmd);
    span.after().skill_output = &out;
    return out;
}

std::string ScriptSkill::exec_(const std::string& cmd)
{
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe)
    {
        throw LCError("ScriptSkill: failed to execute: " + cmd);
    }

    std::string result;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        result += buffer;
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return result;
}

// ---------------- skill_to_tool ----------------

tool::ToolPtr skill_to_tool(SkillPtr skill)
{
    if (!skill)
    {
        return nullptr;
    }

    json schema = {
        {"type", "object"},
        {"properties", json::object()},
        {"required", json::array()}
    };

    return std::make_shared<tool::FunctionTool>(
        skill->name(),
        skill->description(),
        std::move(schema),
        [skill](const json& args) -> std::string
        {
            SkillContext ctx;
            for (auto it = args.begin(); it != args.end(); ++it)
            {
                if (it.value().is_string())
                {
                    ctx.vars[it.key()] = it.value().get<std::string>();
                }
                else
                {
                    ctx.vars[it.key()] = it.value().dump();
                }
            }
            return skill->invoke(ctx);
        });
}

} // namespace skill
} // namespace langchain
