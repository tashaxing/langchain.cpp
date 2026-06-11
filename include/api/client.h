// langchain/api/client.h
// Generic HTTP client for calling any REST API (external AI services and
// framework-built agent services).
//
// Design:
//   - Low-level:  send any HTTP method to any path, with custom headers/body.
//   - Mid-level:  convenience helpers for JSON-based POST/GET/PUT/DELETE.
//   - High-level: OpenAI-compatible chat completions (non-stream + SSE stream).
//
// Works with external providers (OpenAI, DeepSeek, Ollama, etc.) and with
// services built on top of ApiServer (framework agents).
#pragma once

#include "llm/llm.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace langchain
{
namespace api
{

// ---------------------------------------------------------------------------
// HTTP response wrapper.
// ---------------------------------------------------------------------------
struct HttpResponse
{
    int status = 0;                         // HTTP status code (0 = network error)
    std::string body;                       // raw response body
    std::unordered_map<std::string, std::string> headers;

    bool ok() const { return status >= 200 && status < 300; }

    // Parse body as JSON. Returns empty json on parse failure.
    json json_body() const;
};

// ---------------------------------------------------------------------------
// Configuration for the HTTP client.
// ---------------------------------------------------------------------------
struct HttpClientConfig
{
    // Base URL, e.g. "https://api.openai.com" or "http://localhost:8080".
    std::string base_url;

    // Optional Authorization header: "Bearer <api_key>".
    std::string api_key;

    // Default model (used by high-level chat helpers).
    std::string model = "gpt-4o-mini";

    // Timeouts (seconds).
    int connect_timeout_sec = 15;
    int read_timeout_sec    = 120;

    // Extra headers added to every request.
    std::unordered_map<std::string, std::string> extra_headers;
};

// ---------------------------------------------------------------------------
// Generic HTTP client -- call ANY REST endpoint.
// ---------------------------------------------------------------------------
class HttpClient
{
public:
    explicit HttpClient(HttpClientConfig cfg = {});
    ~HttpClient();

    HttpClient(const HttpClient&)            = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // Access / mutate config (thread-safe).
    HttpClientConfig config() const;
    void set_config(const HttpClientConfig& cfg);

    // Low-level: send raw HTTP request. Returns HttpResponse with status=0 on failure.
    HttpResponse request(const std::string& method,
                         const std::string& path,
                         const std::string& body = {},
                         const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    // Convenience wrappers.
    HttpResponse get(const std::string& path,
                      const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    HttpResponse post(const std::string& path,
                      const std::string& body = {},
                      const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    HttpResponse put(const std::string& path,
                     const std::string& body = {},
                     const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    HttpResponse del(const std::string& path,
                     const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    // JSON wrappers (Content-Type/Accept set automatically).
    HttpResponse json_get(const std::string& path,
                          const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    HttpResponse json_post(const std::string& path,
                           const json& body,
                           const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    HttpResponse json_put(const std::string& path,
                          const json& body,
                          const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    HttpResponse json_del(const std::string& path,
                          const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

    // Streaming POST (SSE). The callback receives raw SSE lines (including "data: ...").
    // Return false from the callback to abort the stream.
    HttpResponse stream_post(const std::string& path,
                             const std::string& body,
                             const std::function<bool(const std::string& line)>& on_line,
                             const std::unordered_map<std::string, std::string>& extra_headers = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// AIClient -- high-level wrapper over HttpClient for OpenAI-compatible APIs.
// ---------------------------------------------------------------------------
//
// Also works with framework-built agent services that expose the same
// /v1/chat/completions endpoint (via ApiServer).
//
// If you need full control (custom headers, non-JSON body, etc.), use
// HttpClient directly.

// Callback for streaming text deltas. Return false to abort.
using StreamChunkCallback = std::function<bool(const std::string& delta)>;

class AIClient
{
public:
    explicit AIClient(HttpClientConfig cfg);
    ~AIClient();

    AIClient(const AIClient&)            = delete;
    AIClient& operator=(const AIClient&) = delete;

    // Access the underlying HttpClient for custom requests.
    HttpClient& http();
    const HttpClient& http() const;

    // Configuration shortcuts.
    HttpClientConfig config() const;
    void set_config(const HttpClientConfig& cfg);

    // -----------------------------------------------------------------------
    // OpenAI-compatible chat completions
    // -----------------------------------------------------------------------

    // Non-streaming chat completion.
    llm::ChatResponse invoke(const llm::ChatRequest& req,
                             const std::string& model = {});

    // Streaming chat completion via SSE. The callback receives text deltas.
    llm::ChatResponse invoke_stream(const llm::ChatRequest& req,
                                    const StreamChunkCallback& on_delta,
                                    const std::string& model = {});

    // Convenience: single-shot prompt (non-streaming).
    std::string complete(const std::string& prompt,
                         const std::string& model = {});

    // Convenience: single-shot prompt (streaming).
    std::string complete_stream(const std::string& prompt,
                                const StreamChunkCallback& on_delta,
                                const std::string& model = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Factory helpers for common providers
// ---------------------------------------------------------------------------

inline HttpClientConfig openai_config(const std::string& api_key,
                                       const std::string& model = "gpt-4o-mini")
{
    HttpClientConfig cfg;
    cfg.base_url = "https://api.openai.com";
    cfg.api_key  = api_key;
    cfg.model    = model;
    return cfg;
}

inline HttpClientConfig deepseek_config(const std::string& api_key,
                                         const std::string& model = "deepseek-chat")
{
    HttpClientConfig cfg;
    cfg.base_url = "https://api.deepseek.com";
    cfg.api_key  = api_key;
    cfg.model    = model;
    return cfg;
}

inline HttpClientConfig ollama_config(const std::string& model = "llama3.2",
                                     const std::string& base_url = "http://localhost:11434")
{
    HttpClientConfig cfg;
    cfg.base_url = base_url;
    cfg.model    = model;
    return cfg;
}

inline HttpClientConfig local_agent_config(const std::string& base_url = "http://localhost:8080",
                                            const std::string& model = {})
{
    HttpClientConfig cfg;
    cfg.base_url = base_url;
    cfg.model    = model;
    return cfg;
}

} // namespace api
} // namespace langchain
