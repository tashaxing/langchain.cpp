// langchain/agent/agent.h
// Two agent flavors:
//   * ReActAgent       — text protocol, works with any LLM
//   * ToolCallingAgent — uses provider-native tool_calls (OpenAI-style)
#pragma once

#include "llm/llm.h"
#include "memory/memory.h"
#include "skill/skill_registry.h"
#include "tool/tool.h"
#include "util/common.h"

#include <memory>
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
namespace agent
{

// Trace of one tool call inside an agent run.
struct AgentStep
{
    std::string thought;     // ReAct only
    std::string tool_name;
    std::string tool_input;  // JSON string or raw text
    std::string observation;
};

struct AgentResult
{
    std::string output;
    std::vector<AgentStep> steps;
    bool finished = true;     // false => hit max_iterations
};

// ---------------------------------------------------------------------------
// Agent streaming events
// ---------------------------------------------------------------------------
// When an agent runs in streaming mode it can emit events for *every* step
// (not just the final answer).  This lets the caller observe the agent's
// reasoning process in real time.

enum class AgentStreamEventType
{
    Thought,      // ReAct reasoning text
    ToolCall,     // Name + input of a tool about to be invoked
    Observation,  // Result returned by a tool
    Answer,       // Final answer text (delta for streaming LLM, full for non-streaming)
    Error,        // Something went wrong (e.g. unknown tool, parse error)
    Done          // Agent finished (convenience marker)
};

struct AgentStreamEvent
{
    AgentStreamEventType type;
    std::string text;        // Primary payload (thought, answer delta, observation, error msg)
    std::string tool_name;   // Valid when type == ToolCall
    std::string tool_input;  // Valid when type == ToolCall
};

// Callback signature for agent-level streaming.  Return false to abort.
using AgentStreamCallback = std::function<bool(const AgentStreamEvent&)>;

struct AgentConfig
{
    int max_iterations = 10;
    std::string system_prompt;

    // Inference parameters forwarded to the LLM on every chat request.
    std::optional<float> temperature;
    std::optional<int>   max_tokens;
    std::optional<float> top_p;
    std::optional<int>   top_k;
    bool                 stream = false;
};

// ----- ReAct (text protocol) -----
class ReActAgent
{
public:
    ReActAgent(llm::LLMPtr llm,
               tool::ToolRegistry tools,
               AgentConfig cfg = {},
               memory::MemoryPtr mem = nullptr);

    // Convenience: build from tools + skills. Skills are exported as tools
    // internally so the agent can invoke them the same way it invokes tools.
    ReActAgent(llm::LLMPtr llm,
               tool::ToolRegistry tools,
               skill::SkillRegistry skills,
               AgentConfig cfg = {},
               memory::MemoryPtr mem = nullptr);

    // Invoke with default inference parameters from AgentConfig.
    AgentResult invoke(const std::string& user_input);

    // Invoke with a full Message (preserves multimodal content_parts).
    // `per_request_memory`, if non-null, overrides the agent's default
    // memory for this invocation only — useful for multi-session servers
    // that maintain per-session memory outside the agent.
    AgentResult invoke(const Message& user_msg,
                       const std::optional<float>& temperature = {},
                       const std::optional<int>&   max_tokens = {},
                       const std::optional<float>& top_p = {},
                       const std::optional<int>&   top_k = {},
                       memory::MemoryPtr           per_request_memory = nullptr);

    // Invoke with per-request inference overrides.  Any non-nullopt field
    // overrides the corresponding AgentConfig default for this call only.
    AgentResult invoke(const std::string& user_input,
                       const std::optional<float>& temperature,
                       const std::optional<int>&   max_tokens,
                       const std::optional<float>& top_p,
                       const std::optional<int>&   top_k,
                       bool                        stream = false);

    // Stream the final answer delta-by-delta.  Tool-call iterations are still
    // non-streaming (the agent needs the full response to parse tool_calls),
    // but the final assistant text is delivered via on_delta.
    AgentResult invoke_stream(const std::string& user_input,
                              const llm::StreamCallback& on_delta,
                              const std::optional<float>& temperature = {},
                              const std::optional<int>&   max_tokens = {},
                              const std::optional<float>& top_p = {},
                              const std::optional<int>&   top_k = {});

    // Rich agent-level streaming: emits events for every step (thought,
    // tool_call, observation, answer) so the caller can observe the full
    // reasoning process in real time.
    AgentResult invoke_stream(const std::string& user_input,
                              const AgentStreamCallback& on_event,
                              const std::optional<float>& temperature = {},
                              const std::optional<int>&   max_tokens = {},
                              const std::optional<float>& top_p = {},
                              const std::optional<int>&   top_k = {});

    // Stream with a full Message (preserves multimodal content_parts).
    // `per_request_memory` — see invoke(Message) for semantics.
    AgentResult invoke_stream(const Message& user_msg,
                              const AgentStreamCallback& on_event,
                              const std::optional<float>& temperature = {},
                              const std::optional<int>&   max_tokens = {},
                              const std::optional<float>& top_p = {},
                              const std::optional<int>&   top_k = {},
                              memory::MemoryPtr           per_request_memory = nullptr);

    void set_hooks(hook::HookManager* mgr)
    {
        hooks_ = mgr;
    }

private:
    std::string build_system_prompt() const;
    AgentResult run_impl(const std::string& user_input,
                         const std::optional<float>& temperature,
                         const std::optional<int>&   max_tokens,
                         const std::optional<float>& top_p,
                         const std::optional<int>&   top_k,
                         const llm::StreamCallback*  on_delta = nullptr,
                         const AgentStreamCallback*  on_event = nullptr);
    // Multimodal-aware variant. The string overload above delegates here.
    // If `mem_override` is non-null, it is used instead of mem_ for this call.
    AgentResult run_impl_msg(const Message& user_msg,
                             const std::optional<float>& temperature,
                             const std::optional<int>&   max_tokens,
                             const std::optional<float>& top_p,
                             const std::optional<int>&   top_k,
                             const llm::StreamCallback*  on_delta = nullptr,
                             const AgentStreamCallback*  on_event = nullptr,
                             memory::MemoryPtr           mem_override = nullptr);

    llm::LLMPtr llm_;
    tool::ToolRegistry tools_;
    AgentConfig cfg_;
    memory::MemoryPtr mem_;
    hook::HookManager* hooks_ = nullptr;
};

// ----- OpenAI-style tool calling -----
class ToolCallingAgent
{
public:
    ToolCallingAgent(llm::LLMPtr llm,
                     tool::ToolRegistry tools,
                     AgentConfig cfg = {},
                     memory::MemoryPtr mem = nullptr);

    // Convenience: build from tools + skills. Skills are exported as tools
    // internally so the agent can invoke them the same way it invokes tools.
    ToolCallingAgent(llm::LLMPtr llm,
                     tool::ToolRegistry tools,
                     skill::SkillRegistry skills,
                     AgentConfig cfg = {},
                     memory::MemoryPtr mem = nullptr);

    // Invoke with default inference parameters from AgentConfig.
    AgentResult invoke(const std::string& user_input);

    // Invoke with a full Message (preserves multimodal content_parts).
    // `per_request_memory`, if non-null, overrides the agent's default
    // memory for this invocation only.
    AgentResult invoke(const Message& user_msg,
                       const std::optional<float>& temperature = {},
                       const std::optional<int>&   max_tokens = {},
                       const std::optional<float>& top_p = {},
                       const std::optional<int>&   top_k = {},
                       memory::MemoryPtr           per_request_memory = nullptr);

    // Invoke with per-request inference overrides.
    AgentResult invoke(const std::string& user_input,
                       const std::optional<float>& temperature,
                       const std::optional<int>&   max_tokens,
                       const std::optional<float>& top_p,
                       const std::optional<int>&   top_k,
                       bool                        stream = false);

    // Stream the final answer delta-by-delta.  Tool-call iterations are still
    // non-streaming (the agent needs the full response to parse tool_calls),
    // but the final assistant text is delivered via on_delta.
    AgentResult invoke_stream(const std::string& user_input,
                              const llm::StreamCallback& on_delta,
                              const std::optional<float>& temperature = {},
                              const std::optional<int>&   max_tokens = {},
                              const std::optional<float>& top_p = {},
                              const std::optional<int>&   top_k = {});

    // Rich agent-level streaming: emits events for every step (tool_call,
    // observation, answer) so the caller can observe the full reasoning
    // process in real time.
    AgentResult invoke_stream(const std::string& user_input,
                              const AgentStreamCallback& on_event,
                              const std::optional<float>& temperature = {},
                              const std::optional<int>&   max_tokens = {},
                              const std::optional<float>& top_p = {},
                              const std::optional<int>&   top_k = {});

    // Stream with a full Message (preserves multimodal content_parts).
    // `per_request_memory` — see invoke(Message) for semantics.
    AgentResult invoke_stream(const Message& user_msg,
                              const AgentStreamCallback& on_event,
                              const std::optional<float>& temperature = {},
                              const std::optional<int>&   max_tokens = {},
                              const std::optional<float>& top_p = {},
                              const std::optional<int>&   top_k = {},
                              memory::MemoryPtr           per_request_memory = nullptr);

    void set_hooks(hook::HookManager* mgr)
    {
        hooks_ = mgr;
    }

private:
    AgentResult run_impl(const std::string& user_input,
                         const std::optional<float>& temperature,
                         const std::optional<int>&   max_tokens,
                         const std::optional<float>& top_p,
                         const std::optional<int>&   top_k,
                         const llm::StreamCallback*  on_delta = nullptr,
                         const AgentStreamCallback*  on_event = nullptr);
    // Multimodal-aware variant. The string overload above delegates here.
    // If `mem_override` is non-null, it is used instead of mem_ for this call.
    AgentResult run_impl_msg(const Message& user_msg,
                             const std::optional<float>& temperature,
                             const std::optional<int>&   max_tokens,
                             const std::optional<float>& top_p,
                             const std::optional<int>&   top_k,
                             const llm::StreamCallback*  on_delta = nullptr,
                             const AgentStreamCallback*  on_event = nullptr,
                             memory::MemoryPtr           mem_override = nullptr);

    llm::LLMPtr llm_;
    tool::ToolRegistry tools_;
    AgentConfig cfg_;
    memory::MemoryPtr mem_;
    hook::HookManager* hooks_ = nullptr;
};

} // namespace agent
} // namespace langchain
