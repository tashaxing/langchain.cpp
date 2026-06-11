// src/hook/hook.cpp — HookManager + FunctionHook + ScopedSpan.
#include "hook/hook.h"
#include "util/logging.h"

#include <algorithm>

namespace langchain
{
namespace hook
{

const char* to_string(Phase p)
{
    switch (p)
    {
    case Phase::BeforeLLM:   return "BeforeLLM";
    case Phase::AfterLLM:    return "AfterLLM";
    case Phase::BeforeAgent: return "BeforeAgent";
    case Phase::AfterAgent:  return "AfterAgent";
    case Phase::BeforeSkill: return "BeforeSkill";
    case Phase::AfterSkill:  return "AfterSkill";
    case Phase::BeforeTool:  return "BeforeTool";
    case Phase::AfterTool:   return "AfterTool";
    case Phase::BeforeApi:   return "BeforeApi";
    case Phase::AfterApi:    return "AfterApi";
    }
    return "?";
}

// ---------------- FunctionHook ----------------

FunctionHook::FunctionHook(std::string name, Fn fn, std::vector<Phase> phases)
    : name_(std::move(name)), fn_(std::move(fn)), phases_(std::move(phases))
{
}

bool FunctionHook::wants(Phase p) const
{
    if (phases_.empty())
    {
        return true;
    }
    return std::find(phases_.begin(), phases_.end(), p) != phases_.end();
}

void FunctionHook::on_event(HookContext& ctx)
{
    if (fn_)
    {
        fn_(ctx);
    }
}

std::string FunctionHook::name() const
{
    return name_;
}

// ---------------- HookManager ----------------

std::string HookManager::new_call_id()
{
    std::lock_guard<std::mutex> lk(mu_);
    return "call-" + std::to_string(++next_id_);
}

void HookManager::add(HookPtr h)
{
    if (!h)
    {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    hooks_.push_back(std::move(h));
}

bool HookManager::remove(const std::string& name)
{
    std::lock_guard<std::mutex> lk(mu_);
    auto it = std::remove_if(hooks_.begin(), hooks_.end(),
        [&](const HookPtr& h) { return h && h->name() == name; });
    if (it == hooks_.end())
    {
        return false;
    }
    hooks_.erase(it, hooks_.end());
    return true;
}

void HookManager::clear()
{
    std::lock_guard<std::mutex> lk(mu_);
    hooks_.clear();
}

std::size_t HookManager::size() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return hooks_.size();
}

void HookManager::add(std::string name, FunctionHook::Fn fn, std::vector<Phase> phases)
{
    add(std::make_shared<FunctionHook>(std::move(name), std::move(fn), std::move(phases)));
}

void HookManager::fire(HookContext& ctx) const
{
    // Snapshot under lock so hooks added/removed mid-fire don't blow up.
    std::vector<HookPtr> snapshot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        snapshot = hooks_;
    }
    for (const auto& h : snapshot)
    {
        if (!h || !h->wants(ctx.phase))
        {
            continue;
        }
        try
        {
            h->on_event(ctx);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("[hook] '{}' threw at {}: {}", h->name(), to_string(ctx.phase), e.what());
        }
        catch (...)
        {
            LOG_ERROR("[hook] '{}' threw unknown at {}", h->name(), to_string(ctx.phase));
        }
    }
}

HookManager& HookManager::global()
{
    static HookManager inst;
    return inst;
}

// ---------------- ScopedSpan ----------------

ScopedSpan::ScopedSpan(HookManager* mgr, HookContext before_ctx, Phase after_phase)
    : mgr_(mgr), start_(std::chrono::steady_clock::now())
{
    // Carry call_id + component into the After context.
    after_ctx_.phase     = after_phase;
    after_ctx_.component = before_ctx.component;
    after_ctx_.call_id   = before_ctx.call_id;

    if (mgr_)
    {
        mgr_->fire(before_ctx);
    }
}

ScopedSpan::~ScopedSpan()
{
    if (fired_ || !mgr_)
    {
        return;
    }
    after_ctx_.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_);
    try
    {
        mgr_->fire(after_ctx_);
    }
    catch (...)
    {
        // fire() already swallows individual hook failures; this guards against
        // any other surprise so destructors stay noexcept-compatible.
    }
    fired_ = true;
}

} // namespace hook
} // namespace langchain
