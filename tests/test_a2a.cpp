// tests/test_a2a.cpp — A2A protocol unit tests.
//
// Covers A2AClient / A2AServer round-trip, task lifecycle, and streaming.
#include <gtest/gtest.h>

#include "a2a/a2a_client.h"
#include "a2a/a2a_server.h"
#include "api/server.h"

#include <thread>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

a2a::AgentCard make_test_card(int port)
{
    a2a::AgentCard c;
    c.name = "test-agent";
    c.url  = "http://127.0.0.1:" + std::to_string(port);
    c.version = "1.0";
    return c;
}

} // namespace

// ---------------------------------------------------------------------------
// A2A types — serialization round-trip
// ---------------------------------------------------------------------------
TEST(A2ATypes, TaskRoundTrip)
{
    a2a::Task t;
    t.id = "task-1";
    t.session_id = "sess-1";
    t.status.state = a2a::TaskState::Working;

    auto j = t.to_json();
    auto t2 = a2a::Task::from_json(j);

    EXPECT_EQ(t2.id, t.id);
    EXPECT_EQ(t2.session_id, t.session_id);
    EXPECT_EQ(t2.status.state, t.status.state);
}

TEST(A2ATypes, MessageRoundTrip)
{
    auto m = a2a::Message::text("user", "hello");
    auto j = m.to_json();
    auto m2 = a2a::Message::from_json(j);

    EXPECT_EQ(m2.role, "user");
    ASSERT_EQ(m2.parts.size(), 1u);
    auto* tp = dynamic_cast<a2a::TextPart*>(m2.parts[0].get());
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->text, "hello");
}

TEST(A2ATypes, AgentCardRoundTrip)
{
    a2a::AgentCard c;
    c.name = "agent-x";
    c.description = "does things";
    c.url = "http://localhost:8080";
    c.version = "2.0";
    c.capabilities.streaming = true;

    auto j = c.to_json();
    auto c2 = a2a::AgentCard::from_json(j);

    EXPECT_EQ(c2.name, c.name);
    EXPECT_EQ(c2.description, c.description);
    EXPECT_EQ(c2.url, c.url);
    EXPECT_EQ(c2.version, c.version);
    EXPECT_TRUE(c2.capabilities.streaming);
}

// ---------------------------------------------------------------------------
// A2AServer / A2AClient round-trip
// ---------------------------------------------------------------------------
TEST(A2A, ServerClientRoundTrip)
{
    int port = 19090;
    a2a::AgentCard card = make_test_card(port);

    a2a::A2AServer server(card,
        [](const a2a::Task& task, auto /*update*/) -> a2a::Task
        {
            a2a::Task result = task;
            result.status.state = a2a::TaskState::Completed;
            return result;
        });

    api::ApiServer api({"127.0.0.1", port});
    server.mount(api);
    api.start();
    if (!api.is_running())
    {
        GTEST_SKIP() << "could not bind port " << port;
    }

    a2a::A2AClient client("http://127.0.0.1:" + std::to_string(port));

    a2a::Task task;
    task.id = "t1";
    auto result = client.send_task(task);

    EXPECT_EQ(result.status.state, a2a::TaskState::Completed);

    api.stop();
}

TEST(A2A, GetTask)
{
    int port = 19091;
    a2a::AgentCard card = make_test_card(port);

    a2a::A2AServer server(card,
        [](const a2a::Task& task, auto) -> a2a::Task
        {
            a2a::Task result = task;
            result.status.state = a2a::TaskState::Completed;
            return result;
        });

    api::ApiServer api({"127.0.0.1", port});
    server.mount(api);
    api.start();
    if (!api.is_running())
    {
        GTEST_SKIP() << "could not bind port " << port;
    }

    a2a::A2AClient client("http://127.0.0.1:" + std::to_string(port));

    a2a::Task task;
    task.id = "t2";
    client.send_task(task);

    auto fetched = client.get_task("t2");
    EXPECT_EQ(fetched.id, "t2");

    api.stop();
}

TEST(A2A, CancelTask)
{
    int port = 19092;
    a2a::AgentCard card = make_test_card(port);

    a2a::A2AServer server(card,
        [](const a2a::Task& task, auto) -> a2a::Task
        {
            a2a::Task result = task;
            result.status.state = a2a::TaskState::Completed;
            return result;
        });

    api::ApiServer api({"127.0.0.1", port});
    server.mount(api);
    api.start();
    if (!api.is_running())
    {
        GTEST_SKIP() << "could not bind port " << port;
    }

    a2a::A2AClient client("http://127.0.0.1:" + std::to_string(port));

    a2a::Task task;
    task.id = "t3";
    client.send_task(task);
    client.cancel_task("t3");

    auto fetched = client.get_task("t3");
    EXPECT_EQ(fetched.status.state, a2a::TaskState::Canceled);

    api.stop();
}

// ---------------------------------------------------------------------------
// AgentCard discovery endpoint
// ---------------------------------------------------------------------------
TEST(A2A, DiscoverEndpoint)
{
    int port = 19093;
    a2a::AgentCard card = make_test_card(port);
    card.name = "discoverable";

    a2a::A2AServer server(card,
        [](const a2a::Task& task, auto) -> a2a::Task { return task; });

    api::ApiServer api({"127.0.0.1", port});
    server.mount(api);
    api.start();
    if (!api.is_running())
    {
        GTEST_SKIP() << "could not bind port " << port;
    }

    auto discovered = a2a::AgentCard::discover("http://127.0.0.1:" + std::to_string(port));
    EXPECT_EQ(discovered.name, "discoverable");

    api.stop();
}
