// langchain/api/server.h
// General-purpose HTTP server with optional OpenAI-compatible routes.
//
// Two usage modes:
//   1) OpenAI proxy -- register_model() mounts /v1/models, /v1/chat/completions,
//      /healthz automatically.
//   2) Custom REST API -- add_route() registers arbitrary handlers against
//      user-defined paths. Path parameters (e.g. /users/:id) and query
//      parameters are parsed automatically.
//
// Both modes support SSE streaming through the framework-level StreamResponse
// abstraction. Users call write_sse_chunk()/done() on the Response object;
// the framework handles the chunked transfer encoding and SSE framing.
// No cpp-httplib types leak through the public API.
//
// Multimodal: the OpenAI-spec `messages[].content` may be either a plain
// string OR an array of `{type: "text"|"image_url", ...}` parts. The server
// accepts both shapes -- when an array is present it serializes the parts back
// into a single string (per-part separated with newlines, image_urls rendered
// as "[image_url: <url>]" stubs) and hands that to ILLM. This is a pragmatic
// pass-through: the backend LLM doesn't decode images today, but the schema
// round-trips correctly so callers built against the OpenAI spec keep working.
//
// All requests fire Before*/After* hooks via the configured HookManager.
#pragma once

#include "agent/agent.h"
#include "hook/hook.h"
#include "llm/llm.h"
#include "memory/memory.h"
#include "util/common.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace langchain
{
namespace api
{

// Abstract HTTP request exposed to custom route handlers.
// Decoupled from cpp-httplib so that consumers of api/server.h never
// transitively include httplib.h.
struct Request
{
    std::string method;                      // "GET", "POST", etc.
    std::string path;                        // raw request path (e.g. "/users/42?active=1")
    std::string body;                        // raw request body
    std::unordered_map<std::string, std::string> headers;

    // Path parameters extracted from :name placeholders (e.g. /users/:id).
    std::unordered_map<std::string, std::string> path_params;

    // Query parameters parsed from the URL (e.g. ?active=1&limit=10).
    std::unordered_map<std::string, std::string> query_params;
};

// Sink interface for SSE streaming. Handlers receive this when streaming is
// enabled on the Response. The framework owns the lifetime; the handler only
// needs to call write() / done().
class StreamSink
{
public:
    virtual ~StreamSink() = default;

    // Write one SSE frame. The framework adds the "data: " prefix and \n\n suffix.
    // Returns false if the client disconnected; the handler should stop.
    virtual bool write(const std::string& data) = 0;

    // Signal completion. After this, no further writes are allowed.
    virtual void done() = 0;
};

// Abstract HTTP response built by custom route handlers.
struct Response
{
    int status = 200;
    std::string body;
    std::unordered_map<std::string, std::string> headers;

    // When true, the framework switches to chunked transfer encoding
    // (Content-Type: text/event-stream) and the handler receives a StreamSink
    // via the streaming callback instead of writing to `body`.
    bool stream = false;

    // If `stream` is true, this callback is invoked with a live StreamSink.
    // The handler should call sink->write() repeatedly and finish with sink->done().
    // The framework sets stream=true automatically when this callback is assigned.
    std::function<void(StreamSink& sink)> stream_handler;

    // Convenience: set Content-Type to application/json and body to json::dump().
    void set_json(const json& j)
    {
        headers["Content-Type"] = "application/json";
        body = j.dump();
    }

    // Enable SSE streaming for this response. Returns the StreamSink that the
    // handler should use. Must be called from inside the route handler.
    // After enabling, write data via the returned sink and call sink->done()
    // when finished. Do not touch `body` after enabling streaming.
    void enable_streaming(std::function<void(StreamSink& sink)> handler)
    {
        stream = true;
        stream_handler = std::move(handler);
    }
};

using RouteHandler = std::function<void(const Request& req, Response& res)>;

struct ApiConfig
{
    std::string host       = "0.0.0.0";
    int         port       = 8080;
    std::string api_key;                 // if set, require "Authorization: Bearer <key>"
    int         read_timeout_sec  = 120;
    int         write_timeout_sec = 120;
};

// Maps OpenAI-style model ids to local ILLM implementations. Several model ids
// may share a backend (e.g. an alias and a canonical name).
class ApiServer
{
public:
    explicit ApiServer(ApiConfig cfg = {});
    ~ApiServer();

    ApiServer(const ApiServer&)            = delete;
    ApiServer& operator=(const ApiServer&) = delete;

    // Register a backend keyed by model id. The same llm may be registered
    // under multiple ids. This also mounts the OpenAI routes (/v1/models,
    // /v1/chat/completions, /healthz) on the next run().
    void register_model(std::string id, llm::LLMPtr llm);

    // Register an agent that handles chat requests for a model id.
    // When both an agent and an LLM are registered for the same id,
    // the agent takes priority.
    void register_agent(std::string id, std::shared_ptr<agent::ReActAgent> agent);
    void register_agent(std::string id, std::shared_ptr<agent::ToolCallingAgent> agent);

    // Install a hook manager. Hooks fire for every chat request (BeforeLLM,
    // AfterLLM via the backend), every custom route (BeforeApi, AfterApi),
    // and additionally at the API boundary as metadata-tagged events on the
    // same manager.
    void set_hooks(hook::HookManager* mgr);

    // Register a custom route handler. Supports path parameters with :name
    // syntax (e.g. "/users/:id", "/projects/:pid/tasks/:tid").
    //
    // Handlers are matched in the order they were added. The first matching
    // handler wins; if nothing matches, a 404 is returned.
    //
    // To stream SSE from a custom handler:
    //   res.enable_streaming([&](api::StreamSink& sink) {
    //       sink.write("{\"chunk\":1}");
    //       sink.write("{\"chunk\":2}");
    //       sink.done();
    //   });
    void add_route(const std::string& method,
                   const std::string& path,
                   RouteHandler handler);

    // Optional per-request memory resolver. When set, the OpenAI route
    // (`/v1/chat/completions`) loads conversation history from the resolved
    // memory before invoking the agent, and writes the user message plus the
    // assistant reply back to that memory afterwards. The resolver typically
    // looks at a session header (e.g. X-Session-Id) to return a per-session
    // LongTermMemory instance. Returning nullptr disables memory for that
    // request. The application owns the lifetime of returned memories.
    //
    // This keeps session-aware persistence out of the framework — the
    // server stays generic and the application controls isolation policy.
    using MemoryResolver = std::function<memory::MemoryPtr(const Request&)>;
    void set_memory_resolver(MemoryResolver resolver);

    // Run the HTTP server on the calling thread until stop() is called from
    // another thread or the process exits.
    void run();

    // Same as run() but spawns a worker thread; safe to call from main.
    void start();
    void stop();

    // True between start() and stop(); useful for tests.
    bool is_running() const;

    const ApiConfig& config() const
    {
        return cfg_;
    }

private:
    // PIMPL -- keeps the cpp-httplib include out of every TU that pulls api/server.h.
    struct Impl;
    std::unique_ptr<Impl> impl_;

    ApiConfig                                       cfg_;
    mutable std::mutex                              mu_;
    std::unordered_map<std::string, llm::LLMPtr>    models_;
    std::unordered_map<std::string, std::shared_ptr<agent::ReActAgent>>       react_agents_;
    std::unordered_map<std::string, std::shared_ptr<agent::ToolCallingAgent>> tool_agents_;
    hook::HookManager*                              hooks_ = nullptr;
    MemoryResolver                                  memory_resolver_;
    std::thread                                     worker_;

    // Custom route table: key = "METHOD path" (path with :param syntax preserved).
    std::vector<std::pair<std::string, RouteHandler>> routes_;
};

} // namespace api
} // namespace langchain
