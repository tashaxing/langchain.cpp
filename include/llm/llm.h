// langchain/llm/llm.h
// Abstract LLM interface plus the request/response types shared by all backends.
#pragma once

#include "util/common.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
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
namespace llm
{

// Loose mirror of OpenAI function-schema: name/description + JSON-schema params.
struct ToolSchema
{
    std::string name;
    std::string description;
    json parameters; // JSON-schema object
};

struct ChatRequest
{
    std::vector<Message> messages;
    std::optional<float>       temperature;
    std::optional<int>         max_tokens;
    std::optional<float>       top_p;
    std::optional<int>         top_k;
    std::vector<std::string>   stop;
    std::vector<ToolSchema>    tools;       // optional tool/function schemas
    std::optional<std::string> tool_choice; // "auto" / "none" / specific name
    bool                       stream = false;
};

struct Usage
{
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

struct ChatResponse
{
    Message message;            // assistant reply (may carry tool_calls)
    std::string finish_reason;  // "stop" | "tool_calls" | "length" | ...
    Usage usage{};
    std::string model;
};

// Callback invoked per streamed delta. Return false to abort generation.
using StreamCallback = std::function<bool(const std::string& delta)>;

// Async completion callback.  On success `response` is non-null and `error`
// is null; on failure `response` is null and `error` points to the message.
using AsyncCompleteCallback = std::function<void(const ChatResponse* response,
                                                  const std::string* error)>;

// NVI (non-virtual interface) pattern: the public invoke()/invoke_stream() entry
// points fire Before*/After* hooks and then delegate to virtual invoke_impl()/
// invoke_stream_impl() that subclasses override. Lifecycle observability is
// applied uniformly without every backend having to remember to wire it.
class ILLM
{
public:
    virtual ~ILLM() = default;

    ChatResponse invoke(const ChatRequest& req);

    ChatResponse invoke_stream(const ChatRequest& req, const StreamCallback& on_delta);

    // Convenience: feed a single user prompt, return the assistant text.
    std::string complete(const std::string& prompt)
    {
        ChatRequest req;
        req.messages.push_back(Message::user(prompt));
        return invoke(req).message.content;
    }

    // -----------------------------------------------------------------------
    // Async API (additive — does not replace the synchronous methods above).
    // -----------------------------------------------------------------------
    // Fire-and-forget: invokes the LLM on a background thread and calls
    // `on_complete` when finished.  BeforeLLM fires on the caller thread;
    // AfterLLM fires on the background thread.
    void async_invoke(const ChatRequest& req, const AsyncCompleteCallback& on_complete);

    // Streaming variant: `on_delta` is called for each chunk on the background
    // thread; `on_complete` is called once at the end.
    void async_invoke_stream(const ChatRequest& req,
                              const StreamCallback& on_delta,
                              const AsyncCompleteCallback& on_complete);

    virtual std::string name() const = 0;

    // Optional lifecycle observers; if unset, falls back to HookManager::global().
    void set_hooks(hook::HookManager* mgr)
    {
        hooks_ = mgr;
    }
    hook::HookManager* hooks() const
    {
        return hooks_;
    }

protected:
    // Subclasses implement these. Default invoke_stream_impl falls back to the
    // single-shot invoke_impl + one big delta, so backends without true streaming
    // get sensible behavior for free.
    virtual ChatResponse invoke_impl(const ChatRequest& req) = 0;

    virtual ChatResponse invoke_stream_impl(const ChatRequest& req,
                                            const StreamCallback& on_delta)
    {
        auto resp = invoke_impl(req);
        if (!resp.message.content.empty())
        {
            on_delta(resp.message.content);
        }
        return resp;
    }

    // Subclasses may override these to provide truly async I/O (e.g. libcurl-multi).
    // The default implementation dispatches to the synchronous invoke_impl /
    // invoke_stream_impl on a thread-pool worker.
    virtual void async_invoke_impl(const ChatRequest& req,
                                    const AsyncCompleteCallback& on_complete);

    virtual void async_invoke_stream_impl(const ChatRequest& req,
                                           const StreamCallback& on_delta,
                                           const AsyncCompleteCallback& on_complete);

private:
    hook::HookManager* hooks_ = nullptr;
};

using LLMPtr = std::shared_ptr<ILLM>;

} // namespace llm
} // namespace langchain
