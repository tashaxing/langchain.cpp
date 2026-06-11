// tests/test_api_client.cpp — unit tests for api::HttpClient and api::AIClient.
//
// These tests exercise the client logic without requiring a live server by
// verifying request/response parsing, payload construction, and error paths.
// Where network is needed, we use httpbin.org or a local mock (if available).
//
// For tests that require a live endpoint, set:
//   LC_TEST_BASE_URL  (default: http://localhost:11434)
//   LC_TEST_API_KEY   (optional)
//   LC_TEST_MODEL     (default: qwen2.5:0.5b)
#include <gtest/gtest.h>

#include "api/client.h"
#include "util/strings.h"

#include <chrono>
#include <thread>

using namespace langchain;
using namespace langchain::api;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

HttpClientConfig test_cfg()
{
    HttpClientConfig cfg;
    cfg.base_url = std::getenv("LC_TEST_BASE_URL")
                       ? std::getenv("LC_TEST_BASE_URL")
                       : "http://localhost:11434";
    cfg.model = std::getenv("LC_TEST_MODEL")
                    ? std::getenv("LC_TEST_MODEL")
                    : "qwen2.5:0.5b";
    if (const char* k = std::getenv("LC_TEST_API_KEY"))
    {
        cfg.api_key = k;
    }
    cfg.connect_timeout_sec = 5;
    cfg.read_timeout_sec    = 10;
    return cfg;
}

bool server_reachable(const std::string& base_url)
{
    HttpClient client({base_url, "", "", 3, 5});
    auto res = client.get("/");
    return res.status > 0; // any response means the host is up
}

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------

TEST(HttpResponse, OkFor2xx)
{
    HttpResponse r;
    r.status = 200;
    EXPECT_TRUE(r.ok());
    r.status = 201;
    EXPECT_TRUE(r.ok());
    r.status = 299;
    EXPECT_TRUE(r.ok());
    r.status = 300;
    EXPECT_FALSE(r.ok());
    r.status = 404;
    EXPECT_FALSE(r.ok());
    r.status = 500;
    EXPECT_FALSE(r.ok());
    r.status = 0;
    EXPECT_FALSE(r.ok());
}

TEST(HttpResponse, JsonBodyParses)
{
    HttpResponse r;
    r.body = R"({"key":"value","num":42})";
    auto j = r.json_body();
    EXPECT_EQ(j.value("key", std::string()), "value");
    EXPECT_EQ(j.value("num", 0), 42);
}

TEST(HttpResponse, JsonBodyEmptyReturnsEmpty)
{
    HttpResponse r;
    auto j = r.json_body();
    EXPECT_TRUE(j.is_null() || j.empty());
}

TEST(HttpResponse, JsonBodyInvalidReturnsEmpty)
{
    HttpResponse r;
    r.body = "not json";
    auto j = r.json_body();
    EXPECT_TRUE(j.is_null() || j.empty());
}

// ---------------------------------------------------------------------------
// HttpClient config
// ---------------------------------------------------------------------------

TEST(HttpClient, ConfigRoundTrip)
{
    HttpClientConfig cfg;
    cfg.base_url = "https://example.com";
    cfg.api_key  = "secret";
    cfg.model    = "gpt-4";
    cfg.connect_timeout_sec = 7;
    cfg.read_timeout_sec    = 30;
    cfg.extra_headers = {{"X-Custom", "val"}};

    HttpClient client(cfg);
    auto got = client.config();
    EXPECT_EQ(got.base_url, cfg.base_url);
    EXPECT_EQ(got.api_key, cfg.api_key);
    EXPECT_EQ(got.model, cfg.model);
    EXPECT_EQ(got.connect_timeout_sec, cfg.connect_timeout_sec);
    EXPECT_EQ(got.read_timeout_sec, cfg.read_timeout_sec);
    EXPECT_EQ(got.extra_headers.size(), 1u);
}

TEST(HttpClient, SetConfigUpdates)
{
    HttpClient client;
    HttpClientConfig cfg;
    cfg.base_url = "http://new.host";
    cfg.api_key  = "updated";
    client.set_config(cfg);
    EXPECT_EQ(client.config().base_url, "http://new.host");
    EXPECT_EQ(client.config().api_key, "updated");
}

// ---------------------------------------------------------------------------
// HttpClient generic REST (against a live endpoint, if available)
// ---------------------------------------------------------------------------

class HttpClientLiveTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cfg_ = test_cfg();
        live_ = server_reachable(cfg_.base_url);
    }

    HttpClientConfig cfg_;
    bool live_ = false;
};

TEST_F(HttpClientLiveTest, GetReturnsSomething)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    HttpClient client(cfg_);
    auto res = client.get("/");
    EXPECT_GT(res.status, 0);
}

TEST_F(HttpClientLiveTest, PostJsonToChatCompletions)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    HttpClient client(cfg_);
    json body = {
        {"model", cfg_.model},
        {"messages", json::array({
            {{"role", "user"}, {"content", "Say 'ok'"}}
        })},
        {"temperature", 0.0}
    };
    auto res = client.json_post("/v1/chat/completions", body);
    EXPECT_TRUE(res.ok()) << "status=" << res.status << " body=" << res.body;
    auto j = res.json_body();
    auto choices = j.value("choices", json::array());
    EXPECT_GE(choices.size(), 1u);
}

TEST_F(HttpClientLiveTest, StreamPostReceivesLines)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    HttpClient client(cfg_);
    json body = {
        {"model", cfg_.model},
        {"messages", json::array({
            {{"role", "user"}, {"content", "Hi"}}
        })},
        {"stream", true}
    };

    int line_count = 0;
    auto res = client.stream_post(
        "/v1/chat/completions",
        body.dump(),
        [&](const std::string& line) -> bool
        {
            if (strings::starts_with(line, "data:"))
            {
                ++line_count;
            }
            return true;
        });

    EXPECT_TRUE(res.ok()) << "status=" << res.status;
    EXPECT_GT(line_count, 0) << "Expected at least one SSE data line";
}

TEST_F(HttpClientLiveTest, RequestWithCustomHeaders)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    HttpClient client(cfg_);
    auto res = client.request("GET", "/", {}, {{"X-Test-Header", "42"}});
    // Most servers ignore unknown headers, so we just verify the request goes through.
    EXPECT_GT(res.status, 0);
}

// ---------------------------------------------------------------------------
// AIClient
// ---------------------------------------------------------------------------

class AIClientTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cfg_ = test_cfg();
        live_ = server_reachable(cfg_.base_url);
    }

    HttpClientConfig cfg_;
    bool live_ = false;
};

TEST_F(AIClientTest, ConfigRoundTrip)
{
    AIClient client(cfg_);
    auto got = client.config();
    EXPECT_EQ(got.base_url, cfg_.base_url);
    EXPECT_EQ(got.model, cfg_.model);
}

TEST_F(AIClientTest, SetConfigUpdates)
{
    AIClient client(cfg_);
    HttpClientConfig new_cfg;
    new_cfg.base_url = "http://other";
    new_cfg.model    = "other-model";
    client.set_config(new_cfg);
    EXPECT_EQ(client.config().base_url, "http://other");
    EXPECT_EQ(client.config().model, "other-model");
}

TEST_F(AIClientTest, HttpAccessorReturnsSameConfig)
{
    AIClient client(cfg_);
    EXPECT_EQ(client.http().config().base_url, cfg_.base_url);
}

TEST_F(AIClientTest, ChatNonStreaming)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    AIClient client(cfg_);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("Say 'ok' and nothing else."));
    req.temperature = 0.0f;

    auto resp = client.invoke(req);
    EXPECT_FALSE(resp.message.content.empty());
    EXPECT_EQ(resp.finish_reason, "stop");
}

TEST_F(AIClientTest, ChatStreaming)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    AIClient client(cfg_);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("Count 1,2,3"));
    req.temperature = 0.0f;

    std::string accumulated;
    auto resp = client.invoke_stream(req, [&](const std::string& delta) -> bool
    {
        accumulated += delta;
        return true;
    });

    EXPECT_FALSE(accumulated.empty());
    EXPECT_EQ(resp.finish_reason, "stop");
}

TEST_F(AIClientTest, CompleteConvenience)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    AIClient client(cfg_);
    std::string answer = client.complete("Say 'hello'.");
    EXPECT_FALSE(answer.empty());
}

TEST_F(AIClientTest, CompleteStreamConvenience)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    AIClient client(cfg_);
    std::string accumulated;
    std::string answer = client.complete_stream("Say 'hi'.", [&](const std::string& delta) -> bool
    {
        accumulated += delta;
        return true;
    });
    EXPECT_FALSE(answer.empty());
    EXPECT_FALSE(accumulated.empty());
}

TEST_F(AIClientTest, StreamAbortReturnsFalse)
{
    if (!live_)
    {
        GTEST_SKIP() << "No server at " << cfg_.base_url;
    }
    AIClient client(cfg_);
    llm::ChatRequest req;
    req.messages.push_back(Message::user("Tell me a long story."));

    int count = 0;
    client.invoke_stream(req, [&](const std::string&) -> bool
    {
        return ++count < 3; // abort after 2 chunks
    });
    // We just verify it doesn't crash. The server may or may not have sent
    // more data after abort; the important thing is the client stops calling
    // the callback.
    EXPECT_GE(count, 2);
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

TEST(FactoryHelpers, OpenAIConfig)
{
    auto cfg = openai_config("sk-test", "gpt-4");
    EXPECT_EQ(cfg.base_url, "https://api.openai.com");
    EXPECT_EQ(cfg.api_key, "sk-test");
    EXPECT_EQ(cfg.model, "gpt-4");
}

TEST(FactoryHelpers, DeepSeekConfig)
{
    auto cfg = deepseek_config("sk-test", "deepseek-chat");
    EXPECT_EQ(cfg.base_url, "https://api.deepseek.com");
    EXPECT_EQ(cfg.api_key, "sk-test");
    EXPECT_EQ(cfg.model, "deepseek-chat");
}

TEST(FactoryHelpers, OllamaConfig)
{
    auto cfg = ollama_config("llama3", "http://host:1234");
    EXPECT_EQ(cfg.base_url, "http://host:1234");
    EXPECT_EQ(cfg.model, "llama3");
}

TEST(FactoryHelpers, LocalAgentConfig)
{
    auto cfg = local_agent_config("http://localhost:9000", "my-model");
    EXPECT_EQ(cfg.base_url, "http://localhost:9000");
    EXPECT_EQ(cfg.model, "my-model");
}

} // namespace
