// langchain/hook/hook.h
// Lifecycle hooks — middleware-style insertion points fired before/after every
// LLM call, agent run, skill run, and tool invocation.
//
// Inspired by LangChain's CallbackManager: a hook observes (and may mutate) the
// in-flight request/response, can record metrics, redact PII, short-circuit a
// call, or fan an event out to an external bus. Hooks are owned by a
// HookManager — typically one per ApiServer / agent / LLM — so different
// pipelines can wire different observers.
//
// Hook callbacks must be cheap and exception-safe. Exceptions thrown from a
// hook are caught and logged by HookManager; they do not abort the wrapped
// call.
#pragma once

#include "llm/llm.h"
#include "util/common.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace langchain
{
namespace hook
{

enum class Phase
{
    BeforeLLM,
    AfterLLM,
    BeforeAgent,
    AfterAgent,
    BeforeSkill,
    AfterSkill,
    BeforeTool,
    AfterTool,
    BeforeApi,
    AfterApi
};

const char* to_string(Phase p);

// Context object passed to every hook. Pointers are nullable — only the fields
// relevant to the current phase are populated.
//
// `mutable_*` pointers let a Before* hook rewrite the request in place
// (e.g. inject a system prompt, redact tokens). After* hooks receive a
// non-null response/result pointer.
struct HookContext
{
    Phase       phase;
    std::string component;   // "HttpLLM", "ReActAgent", "PromptSkill", tool name…
    std::string call_id;     // opaque correlation id across Before/After of one call

    // LLM
    llm::ChatRequest*       mutable_request  = nullptr;
    const llm::ChatResponse* response        = nullptr;

    // Agent
    const std::string*  agent_input  = nullptr;
    const std::string*  agent_output = nullptr;

    // Skill
    const std::string*  skill_input  = nullptr;  // serialized vars / question
    const std::string*  skill_output = nullptr;

    // Tool
    const std::string*  tool_name        = nullptr;
    const json*         tool_arguments   = nullptr;
    const std::string*  tool_observation = nullptr;

    // Wall-clock elapsed time for the After* phases; 0 in Before*.
    std::chrono::microseconds elapsed{0};

    // Free-form metadata. Hooks may read/write to share state across phases
    // (keyed by call_id externally if needed).
    json metadata = json::object();
};

class IHook
{
public:
    virtual ~IHook() = default;

    // True if this hook cares about `phase`. Default: handle everything.
    virtual bool wants(Phase) const
    {
        return true;
    }

    virtual void on_event(HookContext& ctx) = 0;

    virtual std::string name() const
    {
        return "anonymous";
    }
};

using HookPtr = std::shared_ptr<IHook>;

// Lambda-backed hook for the common case where a class is overkill.
class FunctionHook : public IHook
{
public:
    using Fn = std::function<void(HookContext&)>;

    FunctionHook(std::string name, Fn fn, std::vector<Phase> phases = {});

    bool wants(Phase p) const override;
    void on_event(HookContext& ctx) override;
    std::string name() const override;

private:
    std::string        name_;
    Fn                 fn_;
    std::vector<Phase> phases_; // empty => all
};

// Thread-safe container of hooks. Fires synchronously on the calling thread;
// exceptions from a hook are swallowed and logged.
class HookManager
{
public:
    HookManager() = default;

    // Mint a fresh correlation id (monotonic per manager). Useful for matching
    // Before* and After* of a single call.
    std::string new_call_id();

    void add(HookPtr h);
    bool remove(const std::string& name);
    void clear();
    std::size_t size() const;

    // Convenience overload — wraps a lambda in a FunctionHook.
    void add(std::string name, FunctionHook::Fn fn, std::vector<Phase> phases = {});

    // Fire `phase` against every interested hook. Caller fills ctx.
    void fire(HookContext& ctx) const;

    // Process-wide default manager. Components that don't get an explicit
    // manager from their owner fall back to this one so users can register
    // app-wide observers in one place.
    static HookManager& global();

private:
    mutable std::mutex      mu_;
    std::vector<HookPtr>    hooks_;
    mutable std::uint64_t   next_id_{0};
};

// ---- helpers used by component impls to keep the boilerplate down ----

// RAII span: captures start time at construction, fires the After* phase on
// destruction with elapsed populated. Used like:
//
//   auto span = ScopedSpan::for_llm(mgr, "HttpLLM", &request);
//   ... call ...
//   span.set_response(&response);   // optional, before scope ends
//
class ScopedSpan
{
public:
    ScopedSpan(HookManager* mgr, HookContext before_ctx, Phase after_phase);
    ~ScopedSpan();

    ScopedSpan(const ScopedSpan&) = delete;
    ScopedSpan& operator=(const ScopedSpan&) = delete;

    // Mutate the After context before it fires (e.g. attach response pointer).
    HookContext& after()
    {
        return after_ctx_;
    }

    const std::string& call_id() const
    {
        return after_ctx_.call_id;
    }

private:
    HookManager*  mgr_;
    HookContext   after_ctx_;
    std::chrono::steady_clock::time_point start_;
    bool          fired_ = false;
};

} // namespace hook
} // namespace langchain
