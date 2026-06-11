#include "agent/agent.h"
#include "hook/hook.h"
#include "util/logging.h"
#include "util/strings.h"

#include <sstream>

namespace langchain
{
namespace agent
{

namespace
{

// ---- ReAct text-protocol parser ----
// Looks for the canonical block:
//   Thought: ...
//   Action: <tool_name>
//   Action Input: <json or text>
// or
//   Final Answer: <text>
struct ReActParse
{
    bool is_final = false;
    std::string thought;
    std::string action;
    std::string action_input;
    std::string final_answer;
};

ReActParse parse_react(const std::string& text)
{
    ReActParse out;
    auto find_after = [&](const std::string& key) -> std::string
    {
        auto pos = text.find(key);
        if (pos == std::string::npos)
        {
            return std::string();
        }
        pos += key.size();
        auto end = text.find('\n', pos);
        std::string v = text.substr(pos,
            end == std::string::npos ? std::string::npos : end - pos);
        return strings::trim(v);
    };

    auto final_pos = text.find("Final Answer:");
    if (final_pos != std::string::npos)
    {
        out.is_final = true;
        out.final_answer = strings::trim(
            text.substr(final_pos + std::string("Final Answer:").size()));
        return out;
    }
    out.thought = find_after("Thought:");
    out.action  = find_after("Action:");
    auto ai_pos = text.find("Action Input:");
    if (ai_pos != std::string::npos)
    {
        out.action_input = strings::trim(
            text.substr(ai_pos + std::string("Action Input:").size()));
    }
    return out;
}

std::string render_tool_catalog(const tool::ToolRegistry& tools)
{
    std::ostringstream oss;
    for (const auto& name : tools.names())
    {
        auto t = tools.get(name);
        oss << "- " << t->name() << ": " << t->description()
            << " | parameters_schema=" << t->parameters_schema().dump() << "\n";
    }
    return oss.str();
}

} // namespace

// ---------------- ReActAgent ----------------

ReActAgent::ReActAgent(llm::LLMPtr llm,
                       tool::ToolRegistry tools,
                       AgentConfig cfg,
                       memory::MemoryPtr mem)
    : llm_(std::move(llm)),
      tools_(std::move(tools)),
      cfg_(std::move(cfg)),
      mem_(std::move(mem))
{
}

ReActAgent::ReActAgent(llm::LLMPtr llm,
                       tool::ToolRegistry tools,
                       skill::SkillRegistry skills,
                       AgentConfig cfg,
                       memory::MemoryPtr mem)
    : llm_(std::move(llm)),
      cfg_(std::move(cfg)),
      mem_(std::move(mem))
{
    tools_.merge(tools);
    tools_.merge(skills.as_tools());
}

std::string ReActAgent::build_system_prompt() const
{
    std::ostringstream oss;
    if (!cfg_.system_prompt.empty())
    {
        oss << cfg_.system_prompt << "\n\n";
    }
    oss << "You are a helpful agent that can call tools to answer questions.\n"
        << "Available tools:\n" << render_tool_catalog(tools_) << "\n"
        << "Use this exact format on each step:\n"
        << "Thought: <your reasoning>\n"
        << "Action: <tool_name>\n"
        << "Action Input: <JSON object matching the tool schema>\n\n"
        << "When you have the final answer, respond with:\n"
        << "Thought: <reasoning>\n"
        << "Final Answer: <answer to the user>\n";
    return oss.str();
}

AgentResult ReActAgent::invoke(const std::string& user_input)
{
    return invoke(user_input,
                  cfg_.temperature,
                  cfg_.max_tokens,
                  cfg_.top_p,
                  cfg_.top_k,
                  cfg_.stream);
}

AgentResult ReActAgent::invoke(const std::string& user_input,
                               const std::optional<float>& temperature,
                               const std::optional<int>&   max_tokens,
                               const std::optional<float>& top_p,
                               const std::optional<int>&   top_k,
                               bool                        /*stream*/)
{
    return run_impl(user_input, temperature, max_tokens, top_p, top_k);
}

AgentResult ReActAgent::invoke_stream(const std::string& user_input,
                                      const llm::StreamCallback& on_delta,
                                      const std::optional<float>& temperature,
                                      const std::optional<int>&   max_tokens,
                                      const std::optional<float>& top_p,
                                      const std::optional<int>&   top_k)
{
    auto result = run_impl(user_input, temperature, max_tokens, top_p, top_k, &on_delta);
    return result;
}

AgentResult ReActAgent::invoke_stream(const std::string& user_input,
                                      const AgentStreamCallback& on_event,
                                      const std::optional<float>& temperature,
                                      const std::optional<int>&   max_tokens,
                                      const std::optional<float>& top_p,
                                      const std::optional<int>&   top_k)
{
    return run_impl(user_input, temperature, max_tokens, top_p, top_k, nullptr, &on_event);
}

// ---- ReActAgent: new Message-based invoke ----

AgentResult ReActAgent::invoke(const Message& user_msg,
                                const std::optional<float>& temperature,
                                const std::optional<int>&   max_tokens,
                                const std::optional<float>& top_p,
                                const std::optional<int>&   top_k,
                                memory::MemoryPtr           per_request_memory)
{
    return run_impl_msg(user_msg,
                        temperature.has_value() ? temperature : cfg_.temperature,
                        max_tokens.has_value()  ? max_tokens  : cfg_.max_tokens,
                        top_p.has_value()       ? top_p       : cfg_.top_p,
                        top_k.has_value()       ? top_k       : cfg_.top_k,
                        nullptr, nullptr,
                        std::move(per_request_memory));
}

AgentResult ReActAgent::invoke_stream(const Message& user_msg,
                                       const AgentStreamCallback& on_event,
                                       const std::optional<float>& temperature,
                                       const std::optional<int>&   max_tokens,
                                       const std::optional<float>& top_p,
                                       const std::optional<int>&   top_k,
                                       memory::MemoryPtr           per_request_memory)
{
    return run_impl_msg(user_msg, temperature, max_tokens, top_p, top_k,
                        nullptr, &on_event, std::move(per_request_memory));
}

// ---- ReActAgent run_impl (string overload delegates to Message overload) ----
AgentResult ReActAgent::run_impl(const std::string& user_input,
                                 const std::optional<float>& temperature,
                                 const std::optional<int>&   max_tokens,
                                 const std::optional<float>& top_p,
                                 const std::optional<int>&   top_k,
                                 const llm::StreamCallback*  on_delta,
                                 const AgentStreamCallback*  on_event)
{
    return run_impl_msg(Message::user(user_input),
                        temperature, max_tokens, top_p, top_k,
                        on_delta, on_event);
}

// Multimodal-aware core: keeps the full user Message (incl. content_parts) so
// that images carried as base64 reach the LLM and are persisted to memory.
AgentResult ReActAgent::run_impl_msg(const Message& user_msg,
                                     const std::optional<float>& temperature,
                                     const std::optional<int>&   max_tokens,
                                     const std::optional<float>& top_p,
                                     const std::optional<int>&   top_k,
                                     const llm::StreamCallback*  on_delta,
                                     const AgentStreamCallback*  on_event,
                                     memory::MemoryPtr           mem_override)
{
    // Hook context wants a const std::string*. Use the plain-text projection
    // of the user message so observability still sees something useful even
    // when the input is multimodal.
    const std::string& user_input = user_msg.content;

    // Per-request memory overrides the agent's default (mem_) so a single
    // shared Agent can serve multiple sessions when used behind ApiServer.
    memory::IMemory* mem = mem_override ? mem_override.get() : mem_.get();

    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "ReActAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_input;
    hook::ScopedSpan agent_span(mgr, before, hook::Phase::AfterAgent);

    AgentResult result;
    std::vector<Message> conv;
    conv.push_back(Message::system(build_system_prompt()));
    if (mem)
    {
        for (const auto& m : mem->messages())
        {
            conv.push_back(m);
        }
    }
    // Push the full user message verbatim so multimodal content_parts survive
    // all the way to the LLM backend.
    conv.push_back(user_msg);

    for (int iter = 0; iter < cfg_.max_iterations; ++iter)
    {
        llm::ChatRequest req;
        req.messages = conv;
        req.stop = {"\nObservation:"};
        req.temperature = temperature;
        req.max_tokens  = max_tokens;
        req.top_p       = top_p;
        req.top_k       = top_k;

        // If streaming is requested, use invoke_stream so the answer is
        // delivered token-by-token.  ReAct still needs the full text to
        // parse whether this step is final or a tool call, so we aggregate
        // locally while forwarding deltas to the caller.
        bool can_stream = (on_delta != nullptr);
        bool can_event  = (on_event != nullptr);

        std::string text;
        if (can_stream)
        {
            std::string aggregate;
            llm_->invoke_stream(req,
                [&](const std::string& delta) -> bool
            {
                aggregate += delta;
                return (*on_delta)(delta);
            });
            text = aggregate;
        }
        else if (can_event)
        {
            // For rich event streaming we still need the full text to parse,
            // but we can stream the raw LLM deltas as "thought" events so the
            // caller sees the reasoning unfold in real time.
            std::string aggregate;
            llm_->invoke_stream(req,
                [&](const std::string& delta) -> bool
            {
                aggregate += delta;
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::Thought;
                ev.text = delta;
                return (*on_event)(ev);
            });
            text = aggregate;
        }
        else
        {
            auto resp = llm_->invoke(req);
            text = resp.message.content;
        }
        conv.push_back(Message::assistant(text));

        auto parse = parse_react(text);
        if (parse.is_final)
        {
            result.output = parse.final_answer;
            result.finished = true;
            if (mem)
            {
                mem->add(user_msg);
                mem->add(Message::assistant(result.output));
            }
            agent_span.after().agent_output = &result.output;

            if (can_event)
            {
                // The final answer text has already been streamed incrementally
                // as Thought events, so we only emit a lightweight Answer marker
                // (no text) to signal the end of reasoning.  This avoids sending
                // the full answer a second time.
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::Answer;
                // ev.text intentionally left empty — deltas were already sent
                (*on_event)(ev);

                AgentStreamEvent done_ev;
                done_ev.type = AgentStreamEventType::Done;
                (*on_event)(done_ev);
            }
            return result;
        }

        AgentStep step;
        step.thought    = parse.thought;
        step.tool_name  = parse.action;
        step.tool_input = parse.action_input;

        if (can_event)
        {
            AgentStreamEvent ev;
            ev.type = AgentStreamEventType::ToolCall;
            ev.tool_name = step.tool_name;
            ev.tool_input = step.tool_input;
            (*on_event)(ev);
        }

        auto t = tools_.get(parse.action);
        if (!t)
        {
            step.observation = "error: unknown tool '" + parse.action + "'";
            if (can_event)
            {
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::Error;
                ev.text = step.observation;
                (*on_event)(ev);
            }
        }
        else
        {
            json args = json::parse(parse.action_input, nullptr, false);
            if (args.is_discarded())
            {
                // Some models drop the braces. Try wrapping bare text under a single
                // schema property if there is exactly one.
                auto schema = t->parameters_schema();
                if (schema.contains("properties") && schema["properties"].size() == 1)
                {
                    const std::string key = schema["properties"].begin().key();
                    args = json::object();
                    args[key] = parse.action_input;
                }
                else
                {
                    args = json::object();
                }
            }

            hook::HookContext tool_before;
            tool_before.phase           = hook::Phase::BeforeTool;
            tool_before.component       = t->name();
            tool_before.call_id         = mgr->new_call_id();
            tool_before.tool_name       = &step.tool_name;
            tool_before.tool_arguments  = &args;
            hook::ScopedSpan tool_span(mgr, tool_before, hook::Phase::AfterTool);

            try
            {
                step.observation = t->invoke(args);
            }
            catch (const std::exception& e)
            {
                step.observation = std::string("error: ") + e.what();
                if (can_event)
                {
                    AgentStreamEvent ev;
                    ev.type = AgentStreamEventType::Error;
                    ev.text = step.observation;
                    (*on_event)(ev);
                }
            }
            tool_span.after().tool_name        = &step.tool_name;
            tool_span.after().tool_observation = &step.observation;
        }
        result.steps.push_back(step);

        if (can_event)
        {
            AgentStreamEvent ev;
            ev.type = AgentStreamEventType::Observation;
            ev.text = step.observation;
            (*on_event)(ev);
        }

        std::ostringstream obs;
        obs << "Observation: " + step.observation;
        conv.push_back(Message::user(obs.str()));
    }

    result.finished = false;
    result.output = "(agent stopped after max_iterations)";
    agent_span.after().agent_output = &result.output;

    if (on_event != nullptr)
    {
        AgentStreamEvent ev;
        ev.type = AgentStreamEventType::Error;
        ev.text = result.output;
        (*on_event)(ev);
    }
    return result;
}

// ---------------- ToolCallingAgent ----------------

ToolCallingAgent::ToolCallingAgent(llm::LLMPtr llm,
                                   tool::ToolRegistry tools,
                                   AgentConfig cfg,
                                   memory::MemoryPtr mem)
    : llm_(std::move(llm)),
      tools_(std::move(tools)),
      cfg_(std::move(cfg)),
      mem_(std::move(mem))
{
}

ToolCallingAgent::ToolCallingAgent(llm::LLMPtr llm,
                                   tool::ToolRegistry tools,
                                   skill::SkillRegistry skills,
                                   AgentConfig cfg,
                                   memory::MemoryPtr mem)
    : llm_(std::move(llm)),
      cfg_(std::move(cfg)),
      mem_(std::move(mem))
{
    tools_.merge(tools);
    tools_.merge(skills.as_tools());
}

AgentResult ToolCallingAgent::invoke(const std::string& user_input)
{
    return invoke(user_input,
                  cfg_.temperature,
                  cfg_.max_tokens,
                  cfg_.top_p,
                  cfg_.top_k,
                  cfg_.stream);
}

AgentResult ToolCallingAgent::invoke(const std::string& user_input,
                                     const std::optional<float>& temperature,
                                     const std::optional<int>&   max_tokens,
                                     const std::optional<float>& top_p,
                                     const std::optional<int>&   top_k,
                                     bool                        /*stream*/)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "ToolCallingAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_input;
    hook::ScopedSpan agent_span(mgr, before, hook::Phase::AfterAgent);

    AgentResult result = run_impl(user_input, temperature, max_tokens, top_p, top_k);
    agent_span.after().agent_output = &result.output;
    return result;
}

AgentResult ToolCallingAgent::invoke_stream(const std::string& user_input,
                                            const llm::StreamCallback& on_delta,
                                            const std::optional<float>& temperature,
                                            const std::optional<int>&   max_tokens,
                                            const std::optional<float>& top_p,
                                            const std::optional<int>&   top_k)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "ToolCallingAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_input;
    hook::ScopedSpan agent_span(mgr, before, hook::Phase::AfterAgent);

    AgentResult result = run_impl(user_input, temperature, max_tokens, top_p, top_k, &on_delta);
    agent_span.after().agent_output = &result.output;
    return result;
}

AgentResult ToolCallingAgent::invoke_stream(const std::string& user_input,
                                            const AgentStreamCallback& on_event,
                                            const std::optional<float>& temperature,
                                            const std::optional<int>&   max_tokens,
                                            const std::optional<float>& top_p,
                                            const std::optional<int>&   top_k)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "ToolCallingAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_input;
    hook::ScopedSpan agent_span(mgr, before, hook::Phase::AfterAgent);

    AgentResult result = run_impl(user_input, temperature, max_tokens, top_p, top_k, nullptr, &on_event);
    agent_span.after().agent_output = &result.output;
    return result;
}

AgentResult ToolCallingAgent::invoke(const Message& user_msg,
                                      const std::optional<float>& temperature,
                                      const std::optional<int>&   max_tokens,
                                      const std::optional<float>& top_p,
                                      const std::optional<int>&   top_k,
                                      memory::MemoryPtr           per_request_memory)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "ToolCallingAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_msg.content;
    hook::ScopedSpan agent_span(mgr, before, hook::Phase::AfterAgent);

    AgentResult result = run_impl_msg(user_msg,
                                      temperature.has_value() ? temperature : cfg_.temperature,
                                      max_tokens.has_value()  ? max_tokens  : cfg_.max_tokens,
                                      top_p.has_value()       ? top_p       : cfg_.top_p,
                                      top_k.has_value()       ? top_k       : cfg_.top_k,
                                      nullptr, nullptr,
                                      std::move(per_request_memory));
    agent_span.after().agent_output = &result.output;
    return result;
}

AgentResult ToolCallingAgent::invoke_stream(const Message& user_msg,
                                             const AgentStreamCallback& on_event,
                                             const std::optional<float>& temperature,
                                             const std::optional<int>&   max_tokens,
                                             const std::optional<float>& top_p,
                                             const std::optional<int>&   top_k,
                                             memory::MemoryPtr           per_request_memory)
{
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    hook::HookContext before;
    before.phase       = hook::Phase::BeforeAgent;
    before.component   = "ToolCallingAgent";
    before.call_id     = mgr->new_call_id();
    before.agent_input = &user_msg.content;
    hook::ScopedSpan agent_span(mgr, before, hook::Phase::AfterAgent);

    AgentResult result = run_impl_msg(user_msg, temperature, max_tokens, top_p, top_k,
                                      nullptr, &on_event,
                                      std::move(per_request_memory));
    agent_span.after().agent_output = &result.output;
    return result;
}

AgentResult ToolCallingAgent::run_impl(const std::string& user_input,
                                       const std::optional<float>& temperature,
                                       const std::optional<int>&   max_tokens,
                                       const std::optional<float>& top_p,
                                       const std::optional<int>&   top_k,
                                       const llm::StreamCallback*  on_delta,
                                       const AgentStreamCallback*  on_event)
{
    return run_impl_msg(Message::user(user_input),
                        temperature, max_tokens, top_p, top_k,
                        on_delta, on_event);
}

// Multimodal-aware core: keeps the full user Message (incl. content_parts) so
// that images carried as base64 reach the LLM and are persisted to memory.
AgentResult ToolCallingAgent::run_impl_msg(const Message& user_msg,
                                            const std::optional<float>& temperature,
                                            const std::optional<int>&   max_tokens,
                                            const std::optional<float>& top_p,
                                            const std::optional<int>&   top_k,
                                            const llm::StreamCallback*  on_delta,
                                            const AgentStreamCallback*  on_event,
                                            memory::MemoryPtr           mem_override)
{
    // Per-request memory overrides mem_ for this call only.
    memory::IMemory* mem = mem_override ? mem_override.get() : mem_.get();

    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();
    AgentResult result;
    std::vector<Message> conv;
    if (!cfg_.system_prompt.empty())
    {
        conv.push_back(Message::system(cfg_.system_prompt));
    }
    if (mem)
    {
        for (const auto& m : mem->messages())
        {
            conv.push_back(m);
        }
    }
    // Push the full user message verbatim so multimodal content_parts survive
    // all the way to the LLM backend.
    conv.push_back(user_msg);

    auto schemas = tools_.schemas();

    for (int iter = 0; iter < cfg_.max_iterations; ++iter)
    {
        llm::ChatRequest req;
        req.messages = conv;
        req.tools = schemas;
        req.tool_choice = schemas.empty()
                              ? std::optional<std::string>{}
                              : std::optional<std::string>{"auto"};
        req.temperature = temperature;
        req.max_tokens  = max_tokens;
        req.top_p       = top_p;
        req.top_k       = top_k;

        // If streaming is requested and this is the final answer round
        // (no tool_calls expected), use invoke_stream so the answer is
        // delivered token-by-token.
        bool can_stream = (on_delta != nullptr);
        bool can_event  = (on_event != nullptr);

        Message assistant;
        if (can_stream)
        {
            std::string aggregate;
            auto resp = llm_->invoke_stream(req,
                [&](const std::string& delta) -> bool
            {
                aggregate += delta;
                return (*on_delta)(delta);
            });
            assistant = resp.message;
        }
        else if (can_event)
        {
            // For rich event streaming we still need the full response to
            // parse tool_calls, but we stream the raw LLM deltas as "thought"
            // events so the caller sees the reasoning unfold in real time.
            std::string aggregate;
            LOG_DEBUG("ToolCallingAgent iter={} calling invoke_stream", iter);
            auto resp = llm_->invoke_stream(req,
                [&](const std::string& delta) -> bool
            {
                LOG_DEBUG("ToolCallingAgent iter={} delta received: len={} content='{}'", iter, delta.size(), delta);
                aggregate += delta;
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::Thought;
                ev.text = delta;
                return (*on_event)(ev);
            });
            LOG_DEBUG("ToolCallingAgent iter={} invoke_stream complete, aggregate_len={} assistant_content_len={} tool_calls_count={}", iter, aggregate.size(), resp.message.content.size(), resp.message.tool_calls.size());
            assistant = resp.message;

            // Some providers do not stream deltas after tool_calls even when
            // stream=true.  If we got a non-empty final answer but no deltas
            // were emitted, split the full text into small chunks and send
            // them as Thought events to simulate streaming.
            if (assistant.tool_calls.empty() &&
                !assistant.content.empty() &&
                aggregate.empty())
            {
                LOG_DEBUG("ToolCallingAgent iter={} no deltas received, simulating stream with chunks", iter);
                const std::string& text = assistant.content;
                const std::size_t chunk_size = 4;  // characters per chunk
                for (std::size_t i = 0; i < text.size(); i += chunk_size)
                {
                    std::size_t len = std::min(chunk_size, text.size() - i);
                    AgentStreamEvent ev;
                    ev.type = AgentStreamEventType::Thought;
                    ev.text = text.substr(i, len);
                    if (!(*on_event)(ev))
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            auto resp = llm_->invoke(req);
            assistant = resp.message;
        }
        conv.push_back(assistant);

        if (assistant.tool_calls.empty())
        {
            result.output = assistant.content;
            result.finished = true;
            if (mem)
            {
                mem->add(user_msg);
                mem->add(Message::assistant(result.output));
            }

            if (can_event)
            {
                // The final answer text has already been streamed incrementally
                // as Thought events, so we only emit a lightweight Answer marker.
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::Answer;
                // ev.text intentionally left empty — deltas were already sent
                (*on_event)(ev);

                AgentStreamEvent done_ev;
                done_ev.type = AgentStreamEventType::Done;
                (*on_event)(done_ev);
            }
            return result;
        }

        for (const auto& tc : assistant.tool_calls)
        {
            AgentStep step;
            step.tool_name  = tc.name;
            step.tool_input = tc.arguments;

            if (can_event)
            {
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::ToolCall;
                ev.tool_name = step.tool_name;
                ev.tool_input = step.tool_input;
                (*on_event)(ev);
            }

            auto t = tools_.get(tc.name);
            std::string obs;
            if (!t)
            {
                obs = "error: unknown tool '" + tc.name + "'";
                if (can_event)
                {
                    AgentStreamEvent ev;
                    ev.type = AgentStreamEventType::Error;
                    ev.text = obs;
                    (*on_event)(ev);
                }
            }
            else
            {
                json args = json::parse(tc.arguments, nullptr, false);
                if (args.is_discarded())
                {
                    args = json::object();
                }

                hook::HookContext tool_before;
                tool_before.phase          = hook::Phase::BeforeTool;
                tool_before.component      = t->name();
                tool_before.call_id        = mgr->new_call_id();
                tool_before.tool_name      = &step.tool_name;
                tool_before.tool_arguments = &args;
                hook::ScopedSpan tool_span(mgr, tool_before, hook::Phase::AfterTool);

                try
                {
                    obs = t->invoke(args);
                }
                catch (const std::exception& e)
                {
                    obs = std::string("error: ") + e.what();
                    if (can_event)
                    {
                        AgentStreamEvent ev;
                        ev.type = AgentStreamEventType::Error;
                        ev.text = obs;
                        (*on_event)(ev);
                    }
                }
                tool_span.after().tool_name        = &step.tool_name;
                tool_span.after().tool_observation = &obs;
            }
            step.observation = obs;
            result.steps.push_back(step);

            if (can_event)
            {
                AgentStreamEvent ev;
                ev.type = AgentStreamEventType::Observation;
                ev.text = obs;
                (*on_event)(ev);
            }

            Message tool_msg = Message::tool(obs, tc.id);
            tool_msg.name = tc.name;
            conv.push_back(std::move(tool_msg));
        }
    }

    result.finished = false;
    result.output = "(agent stopped after max_iterations)";

    if (on_event != nullptr)
    {
        AgentStreamEvent ev;
        ev.type = AgentStreamEventType::Error;
        ev.text = result.output;
        (*on_event)(ev);
    }
    return result;
}

} // namespace agent
} // namespace langchain
