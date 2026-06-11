// examples/12_api_client.cpp — demonstrate the generic HTTP client (HttpClient)
// and the OpenAI-compatible AI client (AIClient).
//
// This example shows:
//   1. Generic REST calls (GET/POST/PUT/DELETE) to any endpoint.
//   2. OpenAI-compatible chat completions (non-streaming and streaming).
//   3. Connecting to both external AI services and framework-built agent services.
//
// For a live demo against a real endpoint, set env vars:
//   LC_BASE_URL   (default: http://localhost:11434 for Ollama)
//   LC_API_KEY    (optional)
//   LC_MODEL      (default: qwen2.5:0.5b)
//
// Or run the 06_api_server.cpp example first, then point this at localhost:8080.
#include "langchain.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

using namespace langchain;

// ---------------------------------------------------------------------------
// Print a horizontal rule.
// ---------------------------------------------------------------------------
static void hr(const std::string& title)
{
    std::cout << "\n========== " << title << " ==========\n";
}

// ---------------------------------------------------------------------------
// Demo 1: Generic HTTP client -- call any REST endpoint.
// ---------------------------------------------------------------------------
static void demo_http_client(const api::HttpClientConfig& cfg)
{
    api::HttpClient http(cfg);

    hr("HttpClient: generic REST");

    // GET /v1/models (OpenAI-compatible)
    std::cout << "GET /v1/models\n";
    auto res = http.json_get("/v1/models");
    std::cout << "  status: " << res.status << "\n";
    if (res.ok())
    {
        auto j = res.json_body();
        std::cout << "  data count: " << j.value("data", json::array()).size() << "\n";
    }
    else
    {
        std::cout << "  body: " << res.body.substr(0, 200) << "\n";
    }

    // POST /v1/chat/completions with raw JSON
    std::cout << "\nPOST /v1/chat/completions (raw JSON)\n";
    json body = {
        {"model", cfg.model},
        {"messages", json::array({
            {{"role", "user"}, {"content", "Say 'hello from HttpClient' in one word"}}
        })},
        {"temperature", 0.0}
    };
    res = http.json_post("/v1/chat/completions", body);
    std::cout << "  status: " << res.status << "\n";
    if (res.ok())
    {
        auto j = res.json_body();
        auto choices = j.value("choices", json::array());
        if (!choices.empty())
        {
            std::cout << "  reply: "
                      << choices[0].value("message", json::object())
                                         .value("content", std::string("(no content)"))
                      << "\n";
        }
    }
    else
    {
        std::cout << "  body: " << res.body.substr(0, 200) << "\n";
    }

    // Custom headers demo
    std::cout << "\nCustom headers demo\n";
    auto res2 = http.get("/healthz", {{"X-Custom-Header", "demo"}});
    std::cout << "  status: " << res2.status << "\n";
}

// ---------------------------------------------------------------------------
// Demo 2: AIClient -- high-level OpenAI-compatible chat.
// ---------------------------------------------------------------------------
static void demo_ai_client(const api::HttpClientConfig& cfg)
{
    api::AIClient client(cfg);

    hr("AIClient: non-streaming chat");
    try
    {
        llm::ChatRequest req;
        req.messages.push_back(Message::system("You are a concise assistant."));
        req.messages.push_back(Message::user("What is 2+2? Answer in one word."));
        req.temperature = 0.0f;

        auto resp = client.invoke(req);
        std::cout << "model: " << resp.model << "\n";
        std::cout << "content: " << resp.message.content << "\n";
        std::cout << "finish_reason: " << resp.finish_reason << "\n";
        std::cout << "usage: prompt=" << resp.usage.prompt_tokens
                  << " completion=" << resp.usage.completion_tokens
                  << " total=" << resp.usage.total_tokens << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "chat failed: " << e.what() << "\n";
    }

    hr("AIClient: streaming chat");
    try
    {
        llm::ChatRequest req;
        req.messages.push_back(Message::user("Count from 1 to 5."));
        req.temperature = 0.0f;

        std::cout << "streaming response: ";
        auto resp = client.invoke_stream(req, [](const std::string& delta) -> bool
        {
            std::cout << delta << std::flush;
            return true; // keep streaming
        });
        std::cout << "\n[stream done] finish_reason=" << resp.finish_reason << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "chat_stream failed: " << e.what() << "\n";
    }

    hr("AIClient: convenience complete()");
    try
    {
        std::string answer = client.complete("Capital of France? One word.");
        std::cout << "answer: " << answer << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "complete failed: " << e.what() << "\n";
    }

    hr("AIClient: convenience complete_stream()");
    try
    {
        std::cout << "streaming: ";
        std::string answer = client.complete_stream(
            "Say 'hi' slowly.",
            [](const std::string& delta) -> bool
            {
                std::cout << delta << std::flush;
                return true;
            });
        std::cout << "\n[stream done] full text: " << answer << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "complete_stream failed: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Demo 3: Connect to a framework-built agent service.
// ---------------------------------------------------------------------------
static void demo_agent_service()
{
    hr("AIClient: connect to framework agent service");

    // Assume an ApiServer is running on localhost:8080 (see 06_api_server.cpp).
    auto cfg = api::local_agent_config("http://localhost:8080", "local");
    api::AIClient agent(cfg);

    try
    {
        // The agent service exposes the same /v1/chat/completions endpoint.
        auto resp = agent.complete("Hello from AIClient!");
        std::cout << "Agent response: " << resp << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "Agent not running (start 06_api_server.cpp first): " << e.what() << "\n";
    }

    // You can also call custom routes on the agent service via HttpClient.
    api::HttpClient http(cfg);
    try
    {
        auto res = http.json_get("/healthz");
        std::cout << "Health check status: " << res.status << "\n";
        if (res.ok())
        {
            std::cout << "Health body: " << res.body << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Health check failed: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Demo 4: Multimodal -- image URL and local image file.
// ---------------------------------------------------------------------------
static void demo_multimodal(const api::HttpClientConfig& cfg)
{
    hr("AIClient: multimodal chat");

    api::AIClient client(cfg);

    // --- 4a: Remote image URL ---
    try
    {
        std::cout << "--- Remote image URL ---\n";
        llm::ChatRequest req;
        req.messages.push_back(Message::user_with_image(
            "What is in this image? Describe in one sentence.",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"));
        req.temperature = 0.0f;

        auto resp = client.invoke(req);
        std::cout << "content: " << resp.message.content << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "multimodal (url) failed: " << e.what() << "\n";
    }

    // --- 4b: Local image file ---
    // In practice, pass an actual image file path to user_with_image_file().
    try
    {
        std::cout << "\n--- Local image file ---\n";
        std::filesystem::create_directories("build");
        std::string dummy_path = "build/multimodal_demo_dummy.png";
        // For demo purposes, we create the API call without a real image.
        // In real usage:
        //   req.messages.push_back(Message::user_with_image_file(
        //       "Describe this image.", "path/to/photo.png"));
        // mime_type is auto-detected from extension, or pass it explicitly.

        llm::ChatRequest req;
        req.messages.push_back(Message::user_with_image_file(
            "Describe this image.",
            dummy_path));
        // Or with explicit mime_type:
        // req.messages.push_back(Message::user_with_image_file(
        //     "Describe this image.", "path/to/photo.jpg", "image/jpeg"));

        auto resp = client.invoke(req);
        std::cout << "content: " << resp.message.content << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "multimodal (file) failed: " << e.what() << "\n";
    }

    // --- 4c: Base64 image (direct data) ---
    try
    {
        std::cout << "\n--- Base64 image data ---\n";
        // A 1x1 transparent PNG in base64.
        std::string tiny_png_b64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

        llm::ChatRequest req;
        req.messages.push_back(Message::user_with_image_base64(
            "What color is this pixel?",
            tiny_png_b64,
            "image/png"));

        auto resp = client.invoke(req);
        std::cout << "content: " << resp.message.content << "\n";
    }
    catch (const std::exception& e)
    {
        std::cout << "multimodal (base64) failed: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Demo 5: Streaming with raw SSE line access via HttpClient.
// ---------------------------------------------------------------------------
static void demo_raw_stream(const api::HttpClientConfig& cfg)
{
    hr("HttpClient: raw SSE stream");

    api::HttpClient http(cfg);
    json body = {
        {"model", cfg.model},
        {"messages", json::array({
            {{"role", "user"}, {"content", "Count 1,2,3"}}
        })},
        {"stream", true}
    };

    std::cout << "raw SSE lines:\n";
    auto res = http.stream_post(
        "/v1/chat/completions",
        body.dump(),
        [](const std::string& line) -> bool
        {
            // line includes the "data: " prefix
            std::cout << "  " << line << "\n";
            return true;
        });

    std::cout << "stream status: " << res.status << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    api::HttpClientConfig cfg;
    cfg.base_url = std::getenv("LC_BASE_URL") ? std::getenv("LC_BASE_URL")
                                               : "http://localhost:11434";
    cfg.model    = std::getenv("LC_MODEL")    ? std::getenv("LC_MODEL")
                                               : "qwen2.5:0.5b";
    if (const char* k = std::getenv("LC_API_KEY"))
    {
        cfg.api_key = k;
    }

    std::cout << "api_client demo\n"
              << "  base_url: " << cfg.base_url << "\n"
              << "  model:    " << cfg.model << "\n"
              << "  api_key:  " << (cfg.api_key.empty() ? "(none)" : "***") << "\n";

    demo_http_client(cfg);
    demo_ai_client(cfg);
    demo_agent_service();
    demo_multimodal(cfg);
    demo_raw_stream(cfg);

    std::cout << "\nDone.\n";
    return 0;
}
