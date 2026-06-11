// tests/test_multi_agent.cpp — CoordinatorAgent unit tests.
#include <gtest/gtest.h>

#include "agent/multi_agent.h"

using namespace langchain;

namespace
{

agent::AgentResult make_result(const std::string& output, bool finished = true)
{
    agent::AgentResult r;
    r.output = output;
    r.finished = finished;
    return r;
}

} // namespace

TEST(CoordinatorAgent, RunsAgentsSequentially)
{
    agent::CoordinatorAgent coord;
    std::vector<std::string> inputs;

    coord.add_agent({"researcher", "finds facts",
        [&inputs](const std::string& input) -> agent::AgentResult
    {
        inputs.push_back(input);
        return make_result("research notes");
    }});
    coord.add_agent({"writer", "writes final",
        [&inputs](const std::string& input) -> agent::AgentResult
    {
        inputs.push_back(input);
        return make_result("final answer");
    }});

    auto result = coord.invoke("question");

    EXPECT_TRUE(result.finished);
    ASSERT_EQ(result.steps.size(), 2u);
    EXPECT_EQ(result.steps[0].agent_name, "researcher");
    EXPECT_EQ(result.steps[1].agent_name, "writer");
    EXPECT_EQ(inputs[0], "question");
    EXPECT_NE(inputs[1].find("research notes"), std::string::npos);
    EXPECT_NE(result.output.find("final answer"), std::string::npos);
}

TEST(CoordinatorAgent, EmptyTeamThrows)
{
    agent::CoordinatorAgent coord;
    EXPECT_THROW(coord.invoke("question"), LCError);
}

TEST(CoordinatorAgent, AddAgentRequiresName)
{
    agent::CoordinatorAgent coord;
    EXPECT_THROW(coord.add_agent({"", "", [](const std::string&) {
        return make_result("x");
    }}), LCError);
}

TEST(CoordinatorAgent, AddAgentRequiresRunner)
{
    agent::CoordinatorAgent coord;
    EXPECT_THROW(coord.add_agent({"missing", "", {}}), LCError);
}

TEST(CoordinatorAgent, StopsOnExceptionByDefault)
{
    agent::CoordinatorAgent coord;
    coord.add_agent({"bad", "throws", [](const std::string&) -> agent::AgentResult
    {
        throw std::runtime_error("boom");
    }});
    coord.add_agent({"never", "not reached", [](const std::string&) -> agent::AgentResult
    {
        return make_result("never");
    }});

    auto result = coord.invoke("question");

    EXPECT_FALSE(result.finished);
    ASSERT_EQ(result.steps.size(), 1u);
    EXPECT_EQ(result.steps[0].agent_name, "bad");
    EXPECT_NE(result.steps[0].error.find("boom"), std::string::npos);
}

TEST(CoordinatorAgent, ContinuesWhenConfigured)
{
    agent::MultiAgentConfig cfg;
    cfg.stop_on_error = false;
    agent::CoordinatorAgent coord(cfg);

    coord.add_agent({"bad", "throws", [](const std::string&) -> agent::AgentResult
    {
        throw std::runtime_error("boom");
    }});
    coord.add_agent({"good", "runs", [](const std::string&) -> agent::AgentResult
    {
        return make_result("ok");
    }});

    auto result = coord.invoke("question");

    EXPECT_FALSE(result.finished);
    ASSERT_EQ(result.steps.size(), 2u);
    EXPECT_EQ(result.steps[1].agent_name, "good");
    EXPECT_EQ(result.steps[1].result.output, "ok");
}

TEST(CoordinatorAgent, CanDisablePreviousOutputPassing)
{
    agent::MultiAgentConfig cfg;
    cfg.pass_previous_outputs = false;
    agent::CoordinatorAgent coord(cfg);

    std::vector<std::string> inputs;
    coord.add_agent({"a", "", [&inputs](const std::string& input) -> agent::AgentResult
    {
        inputs.push_back(input);
        return make_result("a-out");
    }});
    coord.add_agent({"b", "", [&inputs](const std::string& input) -> agent::AgentResult
    {
        inputs.push_back(input);
        return make_result("b-out");
    }});

    coord.invoke("same input");

    ASSERT_EQ(inputs.size(), 2u);
    EXPECT_EQ(inputs[0], "same input");
    EXPECT_EQ(inputs[1], "same input");
}

TEST(CoordinatorAgent, InvokeAsAgentAdaptsResult)
{
    agent::CoordinatorAgent coord;
    coord.add_agent({"a", "", [](const std::string&) -> agent::AgentResult
    {
        return make_result("a-out");
    }});

    auto result = coord.invoke_as_agent("question");

    EXPECT_TRUE(result.finished);
    ASSERT_EQ(result.steps.size(), 1u);
    EXPECT_EQ(result.steps[0].tool_name, "a");
    EXPECT_NE(result.output.find("a-out"), std::string::npos);
}
