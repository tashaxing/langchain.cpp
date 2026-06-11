// src/llm/llm.cpp — NVI invoke()/invoke_stream() that wrap subclasses with hooks.
#include "llm/llm.h"

#include "hook/hook.h"
#include "util/thread_pool.h"

namespace langchain
{
namespace llm
{

ChatResponse ILLM::invoke(const ChatRequest& req)
{
    ChatRequest local = req;
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();

    hook::HookContext before;
    before.phase           = hook::Phase::BeforeLLM;
    before.component       = name();
    before.call_id         = mgr->new_call_id();
    before.mutable_request = &local;
    hook::ScopedSpan span(mgr, before, hook::Phase::AfterLLM);

    auto resp = invoke_impl(local);
    span.after().response = &resp;
    return resp;
}

ChatResponse ILLM::invoke_stream(const ChatRequest& req, const StreamCallback& on_delta)
{
    ChatRequest local = req;
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();

    hook::HookContext before;
    before.phase           = hook::Phase::BeforeLLM;
    before.component       = name();
    before.call_id         = mgr->new_call_id();
    before.mutable_request = &local;
    before.metadata["stream"] = true;
    hook::ScopedSpan span(mgr, before, hook::Phase::AfterLLM);

    auto resp = invoke_stream_impl(local, on_delta);
    span.after().response = &resp;
    return resp;
}

// ---------------------------------------------------------------------------
// Async API — NVI layer
// ---------------------------------------------------------------------------

void ILLM::async_invoke(const ChatRequest& req, const AsyncCompleteCallback& on_complete)
{
    ChatRequest local = req;
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();

    // BeforeLLM fires on the caller thread.
    hook::HookContext before;
    before.phase           = hook::Phase::BeforeLLM;
    before.component       = name();
    before.call_id         = mgr->new_call_id();
    before.mutable_request = &local;
    mgr->fire(before);

    std::string call_id = before.call_id;

    // Wrap the user's callback so AfterLLM fires on the background thread.
    auto wrapped = [this, mgr, call_id, on_complete](
                       const ChatResponse* resp, const std::string* err)
    {
        hook::HookContext after;
        after.phase      = hook::Phase::AfterLLM;
        after.component  = name();
        after.call_id    = call_id;
        after.response   = resp;
        if (err && !err->empty())
        {
            after.metadata["error"] = *err;
        }
        mgr->fire(after);

        if (on_complete)
        {
            on_complete(resp, err);
        }
    };

    async_invoke_impl(local, wrapped);
}

void ILLM::async_invoke_stream(const ChatRequest& req,
                                const StreamCallback& on_delta,
                                const AsyncCompleteCallback& on_complete)
{
    ChatRequest local = req;
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();

    // BeforeLLM fires on the caller thread.
    hook::HookContext before;
    before.phase           = hook::Phase::BeforeLLM;
    before.component       = name();
    before.call_id         = mgr->new_call_id();
    before.mutable_request = &local;
    before.metadata["stream"] = true;
    mgr->fire(before);

    std::string call_id = before.call_id;

    auto wrapped = [this, mgr, call_id, on_complete](
                       const ChatResponse* resp, const std::string* err)
    {
        hook::HookContext after;
        after.phase      = hook::Phase::AfterLLM;
        after.component  = name();
        after.call_id    = call_id;
        after.response   = resp;
        if (err && !err->empty())
        {
            after.metadata["error"] = *err;
        }
        mgr->fire(after);

        if (on_complete)
        {
            on_complete(resp, err);
        }
    };

    async_invoke_stream_impl(local, on_delta, wrapped);
}

// ---------------------------------------------------------------------------
// Default async_impl — dispatches to the synchronous impl on a thread pool.
// ---------------------------------------------------------------------------

void ILLM::async_invoke_impl(const ChatRequest& req,
                              const AsyncCompleteCallback& on_complete)
{
    if (!util::ThreadPool::default_pool().submit(
        [this, req, on_complete]() mutable
        {
            try
            {
                auto resp = invoke_impl(req);
                if (on_complete)
                {
                    on_complete(&resp, nullptr);
                }
            }
            catch (const std::exception& e)
            {
                if (on_complete)
                {
                    std::string err = e.what();
                    on_complete(nullptr, &err);
                }
            }
            catch (...)
            {
                if (on_complete)
                {
                    std::string err = "unknown exception";
                    on_complete(nullptr, &err);
                }
            }
        }))
    {
        if (on_complete)
        {
            std::string err = "thread pool shutting down";
            on_complete(nullptr, &err);
        }
    }
}

void ILLM::async_invoke_stream_impl(const ChatRequest& req,
                                     const StreamCallback& on_delta,
                                     const AsyncCompleteCallback& on_complete)
{
    if (!util::ThreadPool::default_pool().submit(
        [this, req, on_delta, on_complete]() mutable
        {
            try
            {
                auto resp = invoke_stream_impl(req, on_delta);
                if (on_complete)
                {
                    on_complete(&resp, nullptr);
                }
            }
            catch (const std::exception& e)
            {
                if (on_complete)
                {
                    std::string err = e.what();
                    on_complete(nullptr, &err);
                }
            }
            catch (...)
            {
                if (on_complete)
                {
                    std::string err = "unknown exception";
                    on_complete(nullptr, &err);
                }
            }
        }))
    {
        if (on_complete)
        {
            std::string err = "thread pool shutting down";
            on_complete(nullptr, &err);
        }
    }
}

} // namespace llm
} // namespace langchain
