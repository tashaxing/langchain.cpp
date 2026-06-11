// tests/test_api_server.cpp — ApiServer unit tests.
//
// Brings up the HTTP server on a local port and drives it with HttpClient.
#include <gtest/gtest.h>

#include "api/server.h"
#include "api/client.h"
#include "llm/llm.h"

#include <thread>

using namespace langchain;

namespace
{

// ---------------------------------------------------------------------------
// EchoLLM — mock backend.
// ---------------------------------------------------------------------------
class EchoLLM : public llm::ILLM
{
public:
    std::string name() const override { return "echo"; }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
    {
        llm::ChatResponse out;
        std::string last;
        for (const auto& m : req.messages)
        {
            if (m.role == Role::User) { last = m.content; }
        }
        out.message = Message::assistant("echo: " + last);
        out.finish_reason = "stop";
        return out;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Custom routes
// ---------------------------------------------------------------------------
TEST(ApiServer, CustomGetRoute)
{
    api::ApiServer server({"127.0.0.1", 18080});
    server.add_route("GET", "/hello",
        [](const api::Request&, api::Response& res)
        {
            res.body = "world";
        });
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::HttpClient client({"http://127.0.0.1:18080"});
    auto resp = client.get("/hello");
    EXPECT_TRUE(resp.ok());
    EXPECT_EQ(resp.body, "world");

    server.stop();
}

TEST(ApiServer, CustomPostRoute)
{
    api::ApiServer server({"127.0.0.1", 18081});
    server.add_route("POST", "/echo",
        [](const api::Request& req, api::Response& res)
        {
            res.set_json({{"received", req.body}});
        });
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::HttpClient client({"http://127.0.0.1:18081"});
    auto resp = client.post("/echo", "test-body");
    EXPECT_TRUE(resp.ok());
    EXPECT_NE(resp.body.find("test-body"), std::string::npos);

    server.stop();
}

TEST(ApiServer, PathParams)
{
    api::ApiServer server({"127.0.0.1", 18082});
    server.add_route("GET", "/users/:id",
        [](const api::Request& req, api::Response& res)
        {
            auto it = req.path_params.find("id");
            if (it != req.path_params.end())
            {
                res.body = "user=" + it->second;
            }
            else
            {
                res.status = 400;
            }
        });
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::HttpClient client({"http://127.0.0.1:18082"});
    auto resp = client.get("/users/42");
    EXPECT_TRUE(resp.ok());
    EXPECT_EQ(resp.body, "user=42");

    server.stop();
}

// ---------------------------------------------------------------------------
// OpenAI-compatible routes
// ---------------------------------------------------------------------------
TEST(ApiServer, ModelsEndpoint)
{
    api::ApiServer server({"127.0.0.1", 18083});
    server.register_model("gpt-test", std::make_shared<EchoLLM>());
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::HttpClient client({"http://127.0.0.1:18083"});
    auto resp = client.get("/v1/models");
    EXPECT_TRUE(resp.ok());
    EXPECT_NE(resp.body.find("gpt-test"), std::string::npos);

    server.stop();
}

TEST(ApiServer, ChatCompletionsNonStreaming)
{
    api::ApiServer server({"127.0.0.1", 18084});
    server.register_model("gpt-test", std::make_shared<EchoLLM>());
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::AIClient client({"http://127.0.0.1:18084", "", "gpt-test", 5, 10});
    auto answer = client.complete("hello");
    EXPECT_EQ(answer, "echo: hello");

    server.stop();
}

TEST(ApiServer, CustomRouteExceptionReturns500)
{
    api::ApiServer server({"127.0.0.1", 18086});
    server.add_route("GET", "/boom",
        [](const api::Request&, api::Response&)
        {
            throw std::runtime_error("boom");
        });
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::HttpClient client({"http://127.0.0.1:18086"});
    auto resp = client.get("/boom");
    EXPECT_EQ(resp.status, 500);
    EXPECT_NE(resp.body.find("boom"), std::string::npos);

    server.stop();
}

TEST(ApiServer, StartTwiceThrows)
{
    api::ApiServer server({"127.0.0.1", 18087});
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    EXPECT_THROW(server.start(), LCError);

    server.stop();
}

// ---------------------------------------------------------------------------
// Health endpoint
// ---------------------------------------------------------------------------
TEST(ApiServer, Healthz)
{
    api::ApiServer server({"127.0.0.1", 18085});
    server.start();
    if (!server.is_running())
    {
        GTEST_SKIP() << "could not bind port";
    }

    api::HttpClient client({"http://127.0.0.1:18085"});
    auto resp = client.get("/healthz");
    EXPECT_TRUE(resp.ok());
    EXPECT_NE(resp.body.find("ok"), std::string::npos);

    server.stop();
}
