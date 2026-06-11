// tests/test_agent.cpp — Agent unit tests.
//
// Covers ReActAgent and ToolCallingAgent with a mock EchoLLM so no network
// is required.  Tests invoke, streaming, tool calling, error paths, and
// hook firing.
#include <gtest/gtest.h>

#include "agent/agent.h"
#include "llm/llm.h"
#include "tool/tool.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// EchoLLM — deterministic mock backend.
// Returns ReAct-compatible format so the agent can parse Final Answer.
// ---------------------------------------------------------------------------
class EchoLLM : public llm::ILLM
{
public:
    std::string name() const override
    {
        return "echo";
    }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
    {
        llm::ChatResponse out;
        out.model = "echo";
        std::string last;
        for (const auto& m : req.messages)
        {
            if (m.role == Role::User)
            {
                last = m.content;
            }
        }
        // ReAct format: the agent looks for "Final Answer:".
        out.message = Message::assistant(
            "Thought: I should echo back.\nFinal Answer: echo: " + last);
        out.finish_reason = "stop";
        return out;
    }
};

// ---------------------------------------------------------------------------
// ToolCallLLM — mock that returns tool_calls when tools are present.
// ---------------------------------------------------------------------------
class ToolCallLLM : public llm::ILLM
{
public:
    std::string name() const override
    {
        return "toolcall";
    }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
    {
        llm::ChatResponse out;
        out.model = "toolcall";

        if (!req.tools.empty() && !called_tool_)
        {
            // First call: return a tool_call.
            called_tool_ = true;
            out.message.tool_calls.push_back({
                "call_1",
                "calculator",
                R"({"expression":"2+3"})"
            });
            out.finish_reason = "tool_calls";
        }
        else
        {
            // Second call (after tool result): return final answer.
            out.message = Message::assistant("The answer is 5.");
            out.finish_reason = "stop";
        }
        return out;
    }

private:
    bool called_tool_ = false;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

tool::ToolPtr make_calculator()
{
    return tool::make_calculator_tool();
}

} // namespace

// ---------------------------------------------------------------------------
// ReActAgent — basic invoke
// ---------------------------------------------------------------------------
TEST(ReActAgent, InvokeReturnsOutput)
{
    auto llm = std::make_shared<EchoLLM>();
    tool::ToolRegistry tools;
    agent::ReActAgent agent(llm, std::move(tools));

    auto result = agent.invoke("hello");
    EXPECT_EQ(result.output, "echo: hello");
    EXPECT_TRUE(result.finished);
}

// ---------------------------------------------------------------------------
// ReActAgent — with a real tool
// ---------------------------------------------------------------------------
TEST(ReActAgent, InvokeWithToolCallsTool)
{
    auto llm = std::make_shared<ToolCallLLM>();
    tool::ToolRegistry tools;
    tools.add(make_calculator());
    agent::ReActAgent agent(llm, std::move(tools));

    auto result = agent.invoke("What is 2+3?");
    // The mock ToolCallLLM is OpenAI-style; ReActAgent expects text protocol,
    // so it won't parse tool_calls.  This test verifies it doesn't crash.
    EXPECT_FALSE(result.output.empty());
}

// ---------------------------------------------------------------------------
// ReActAgent — streaming (delta callback)
// ---------------------------------------------------------------------------
TEST(ReActAgent, InvokeStreamDeltas)
{
    auto llm = std::make_shared<EchoLLM>();
    tool::ToolRegistry tools;
    agent::ReActAgent agent(llm, std::move(tools));

    std::string accumulated;
    auto result = agent.invoke_stream("world",
        [&accumulated](const std::string& delta) -> bool
    {
        accumulated += delta;
        return true;
    });

    EXPECT_EQ(result.output, "echo: world");
    // The stream callback receives the full assistant text (including ReAct
    // formatting) because ReAct needs the complete response to parse it.
    EXPECT_NE(accumulated.find("echo: world"), std::string::npos);
}

// ---------------------------------------------------------------------------
// ReActAgent — rich streaming (event callback)
// ---------------------------------------------------------------------------
TEST(ReActAgent, InvokeStreamEvents)
{
    auto llm = std::make_shared<EchoLLM>();
    tool::ToolRegistry tools;
    agent::ReActAgent agent(llm, std::move(tools));

    std::vector<agent::AgentStreamEventType> event_types;
    auto result = agent.invoke_stream("test",
        [&event_types](const agent::AgentStreamEvent& ev) -> bool
    {
        event_types.push_back(ev.type);
        return true;
    });

    EXPECT_FALSE(event_types.empty());
    EXPECT_EQ(event_types.back(), agent::AgentStreamEventType::Done);
}

// ---------------------------------------------------------------------------
// ReActAgent — max_iterations stops loop
// ---------------------------------------------------------------------------
TEST(ReActAgent, MaxIterationsStops)
{
    // An LLM that always returns a tool call (never final answer).
    class AlwaysToolLLM : public llm::ILLM
    {
    public:
        std::string name() const override { return "always-tool"; }
    protected:
        llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
        {
            llm::ChatResponse out;
            out.message.tool_calls.push_back({
                "call_x", "unknown_tool", "{}"
            });
            out.finish_reason = "tool_calls";
            return out;
        }
    };

    auto llm = std::make_shared<AlwaysToolLLM>();
    tool::ToolRegistry tools;
    agent::AgentConfig cfg;
    cfg.max_iterations = 2;
    agent::ReActAgent agent(llm, std::move(tools), cfg);

    auto result = agent.invoke("trigger loop");
    EXPECT_FALSE(result.finished);
    EXPECT_EQ(result.steps.size(), 2u);
}

// ---------------------------------------------------------------------------
// ToolCallingAgent — basic tool invocation
// ---------------------------------------------------------------------------
TEST(ToolCallingAgent, InvokeWithTool)
{
    auto llm = std::make_shared<ToolCallLLM>();
    tool::ToolRegistry tools;
    tools.add(make_calculator());
    agent::ToolCallingAgent agent(llm, std::move(tools));

    auto result = agent.invoke("What is 2+3?");
    EXPECT_EQ(result.output, "The answer is 5.");
    EXPECT_TRUE(result.finished);
    ASSERT_EQ(result.steps.size(), 1u);
    EXPECT_EQ(result.steps[0].tool_name, "calculator");
}

// ---------------------------------------------------------------------------
// ToolCallingAgent — unknown tool returns error observation
// ---------------------------------------------------------------------------
TEST(ToolCallingAgent, UnknownToolReturnsError)
{
    class BadToolLLM : public llm::ILLM
    {
    public:
        std::string name() const override { return "bad-tool"; }
    protected:
        llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
        {
            llm::ChatResponse out;
            out.message.tool_calls.push_back({
                "call_1", "nonexistent", "{}"
            });
            out.finish_reason = "tool_calls";
            return out;
        }
    };

    auto llm = std::make_shared<BadToolLLM>();
    tool::ToolRegistry tools;
    agent::AgentConfig cfg;
    cfg.max_iterations = 1;
    agent::ToolCallingAgent agent(llm, std::move(tools), cfg);

    auto result = agent.invoke("call bad tool");
    ASSERT_EQ(result.steps.size(), 1u);
    EXPECT_NE(result.steps[0].observation.find("unknown tool"), std::string::npos);
}

// ---------------------------------------------------------------------------
// ToolCallingAgent — streaming events
// ---------------------------------------------------------------------------
TEST(ToolCallingAgent, InvokeStreamEvents)
{
    auto llm = std::make_shared<ToolCallLLM>();
    tool::ToolRegistry tools;
    tools.add(make_calculator());
    agent::ToolCallingAgent agent(llm, std::move(tools));

    std::vector<agent::AgentStreamEventType> event_types;
    auto result = agent.invoke_stream("What is 2+3?",
        [&event_types](const agent::AgentStreamEvent& ev) -> bool
    {
        event_types.push_back(ev.type);
        return true;
    });

    EXPECT_FALSE(event_types.empty());
    EXPECT_EQ(event_types.back(), agent::AgentStreamEventType::Done);
}

// ---------------------------------------------------------------------------
// ToolCallingAgent — malformed tool arguments are handled
// ---------------------------------------------------------------------------
TEST(ToolCallingAgent, MalformedArgumentsHandled)
{
    class BadArgsLLM : public llm::ILLM
    {
    public:
        std::string name() const override { return "bad-args"; }
    protected:
        llm::ChatResponse invoke_impl(const llm::ChatRequest&) override
        {
            llm::ChatResponse out;
            out.message.tool_calls.push_back({
                "call_1", "calculator", "not valid json"
            });
            out.finish_reason = "tool_calls";
            return out;
        }
    };

    auto llm = std::make_shared<BadArgsLLM>();
    tool::ToolRegistry tools;
    tools.add(make_calculator());
    agent::ToolCallingAgent agent(llm, std::move(tools));

    // Should not throw; falls back to empty JSON object.
    auto result = agent.invoke("bad args");
    EXPECT_FALSE(result.output.empty());
}
