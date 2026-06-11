// src/api/server.cpp -- OpenAI-compatible REST front-end and custom route
// dispatcher built on cpp-httplib.
#include "api/server.h"

#include "agent/agent.h"
#include "util/logging.h"
#include "util/strings.h"

#include <httplib.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <sstream>

namespace langchain
{
namespace api
{

namespace
{

// --------- Multimodal-aware message decoding ---------
//
// OpenAI's `content` field is either a string or an array of `{type, ...}` parts.
// We parse both shapes into Message::content_parts (multimodal) and also
// flatten to Message::content (text) for backward-compat with text-only backends.
std::string flatten_content(const json& content)
{
    if (content.is_string())
    {
        return content.get<std::string>();
    }
    if (!content.is_array())
    {
        return content.dump();
    }
    std::ostringstream oss;
    bool first = true;
    for (const auto& part : content)
    {
        if (!first)
        {
            oss << "\n";
        }
        first = false;
        std::string type = part.value("type", std::string("text"));
        if (type == "text")
        {
            oss << part.value("text", std::string());
        }
        else if (type == "image_url")
        {
            std::string url;
            if (part.contains("image_url"))
            {
                const auto& iu = part["image_url"];
                if (iu.is_string())
                {
                    url = iu.get<std::string>();
                }
                else if (iu.is_object())
                {
                    url = iu.value("url", std::string());
                }
            }
            oss << "[image_url: " << url << "]";
        }
        else
        {
            // Pass unknown part types through as compact JSON for transparency.
            oss << "[" << type << ": " << part.dump() << "]";
        }
    }
    return oss.str();
}

// Parse a multimodal content array into ContentParts.
std::vector<ContentPart> parse_content_parts(const json& content)
{
    std::vector<ContentPart> parts;
    if (!content.is_array())
    {
        return parts;
    }
    for (const auto& part : content)
    {
        std::string type = part.value("type", std::string("text"));
        if (type == "text")
        {
            parts.push_back(ContentPart::text_part(
                part.value("text", std::string())));
        }
        else if (type == "image_url")
        {
            const auto& iu = part.value("image_url", json::object());
            std::string url = iu.value("url", std::string());
            if (url.find("data:") == 0)
            {
                auto comma = url.find(',');
                if (comma != std::string::npos)
                {
                    std::string prefix = url.substr(0, comma);
                    std::string data = url.substr(comma + 1);
                    std::string mime = "image/png";
                    auto semi = prefix.find(';');
                    if (semi != std::string::npos)
                    {
                        mime = prefix.substr(5, semi - 5);
                    }
                    parts.push_back(ContentPart::image_base64(std::move(data), mime));
                }
                else
                {
                    parts.push_back(ContentPart::image_url(std::move(url)));
                }
            }
            else
            {
                parts.push_back(ContentPart::image_url(std::move(url)));
            }
        }
        else
        {
            parts.push_back(ContentPart::text_part(part.dump()));
        }
    }
    return parts;
}

llm::ChatRequest decode_request(const json& body)
{
    llm::ChatRequest req;
    if (body.contains("messages") && body["messages"].is_array())
    {
        for (const auto& jm : body["messages"])
        {
            Message m;
            m.role = role_from_string(jm.value("role", std::string("user")));
            if (jm.contains("content"))
            {
                const auto& c = jm["content"];
                if (c.is_array())
                {
                    m.content_parts = parse_content_parts(c);
                    m.content = flatten_content(c);
                }
                else if (c.is_string())
                {
                    m.content = c.get<std::string>();
                }
            }
            m.name         = jm.value("name", std::string());
            m.tool_call_id = jm.value("tool_call_id", std::string());
            if (jm.contains("tool_calls") && jm["tool_calls"].is_array())
            {
                for (const auto& tc : jm["tool_calls"])
                {
                    ToolCall c;
                    c.id = tc.value("id", std::string());
                    const auto& fn = tc.value("function", json::object());
                    c.name = fn.value("name", std::string());
                    if (fn.contains("arguments"))
                    {
                        c.arguments = fn["arguments"].is_string()
                                          ? fn["arguments"].get<std::string>()
                                          : fn["arguments"].dump();
                    }
                    m.tool_calls.push_back(std::move(c));
                }
            }
            req.messages.push_back(std::move(m));
        }
    }
    if (body.contains("temperature") && body["temperature"].is_number())
    {
        req.temperature = body["temperature"].get<float>();
    }
    if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())
    {
        req.max_tokens = body["max_tokens"].get<int>();
    }
    if (body.contains("top_p") && body["top_p"].is_number())
    {
        req.top_p = body["top_p"].get<float>();
    }
    if (body.contains("top_k") && body["top_k"].is_number_integer())
    {
        req.top_k = body["top_k"].get<int>();
    }
    if (body.contains("stop"))
    {
        if (body["stop"].is_string())
        {
            req.stop.push_back(body["stop"].get<std::string>());
        }
        else if (body["stop"].is_array())
        {
            for (const auto& s : body["stop"])
            {
                if (s.is_string())
                {
                    req.stop.push_back(s.get<std::string>());
                }
            }
        }
    }
    if (body.contains("tools") && body["tools"].is_array())
    {
        for (const auto& t : body["tools"])
        {
            llm::ToolSchema ts;
            const auto& fn = t.value("function", json::object());
            ts.name        = fn.value("name", std::string());
            ts.description = fn.value("description", std::string());
            ts.parameters  = fn.value("parameters", json::object());
            req.tools.push_back(std::move(ts));
        }
    }
    if (body.contains("tool_choice"))
    {
        if (body["tool_choice"].is_string())
        {
            req.tool_choice = body["tool_choice"].get<std::string>();
        }
        else
        {
            req.tool_choice = body["tool_choice"].dump();
        }
    }
    req.stream = body.value("stream", false);
    return req;
}

json encode_message(const Message& m)
{
    json j;
    j["role"] = to_string(m.role);
    j["content"] = m.content;
    if (!m.tool_calls.empty())
    {
        json arr = json::array();
        for (const auto& tc : m.tool_calls)
        {
            arr.push_back({
                {"id", tc.id},
                {"type", "function"},
                {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
            });
        }
        j["tool_calls"] = std::move(arr);
    }
    return j;
}

json encode_response(const llm::ChatResponse& resp, const std::string& model_id)
{
    json out;
    out["id"]      = "chatcmpl-lc-" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    out["object"]  = "chat.completion";
    out["created"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    out["model"]   = resp.model.empty() ? model_id : resp.model;
    out["choices"] = json::array({
        {
            {"index", 0},
            {"message", encode_message(resp.message)},
            {"finish_reason", resp.finish_reason.empty() ? "stop" : resp.finish_reason}
        }
    });
    out["usage"] = {
        {"prompt_tokens",     resp.usage.prompt_tokens},
        {"completion_tokens", resp.usage.completion_tokens},
        {"total_tokens",      resp.usage.total_tokens}
    };
    return out;
}

std::string make_sse_chunk(const std::string& model_id, const std::string& delta,
                           const std::string& finish_reason)
{
    json choice = {
        {"index", 0},
        {"delta", json::object()},
        {"finish_reason", finish_reason.empty() ? json(nullptr) : json(finish_reason)}
    };
    if (!delta.empty())
    {
        choice["delta"]["content"] = delta;
    }
    json j = {
        {"id",      "chatcmpl-lc-stream"},
        {"object",  "chat.completion.chunk"},
        {"created", std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()},
        {"model",   model_id},
        {"choices", json::array({choice})}
    };
    return "data: " + j.dump() + "\n\n";
}

// SSE chunk for agent-level streaming events.  Extends the OpenAI chunk format
// with an "agent_event" field so the client can distinguish thought, tool_call,
// observation, answer, error, and done events.
std::string make_agent_sse_chunk(const std::string& model_id,
                                 const agent::AgentStreamEvent& ev,
                                 const std::string& finish_reason)
{
    json choice = {
        {"index", 0},
        {"delta", json::object()},
        {"finish_reason", finish_reason.empty() ? json(nullptr) : json(finish_reason)}
    };

    json agent_event = {
        {"type", [&]() -> std::string {
            switch (ev.type)
            {
                case agent::AgentStreamEventType::Thought:     return "thought";
                case agent::AgentStreamEventType::ToolCall:    return "tool_call";
                case agent::AgentStreamEventType::Observation: return "observation";
                case agent::AgentStreamEventType::Answer:      return "answer";
                case agent::AgentStreamEventType::Error:       return "error";
                case agent::AgentStreamEventType::Done:        return "done";
            }
            return "unknown";
        }()}
    };
    if (!ev.text.empty())
    {
        agent_event["text"] = ev.text;
    }
    if (!ev.tool_name.empty())
    {
        agent_event["tool_name"] = ev.tool_name;
    }
    if (!ev.tool_input.empty())
    {
        agent_event["tool_input"] = ev.tool_input;
    }
    choice["agent_event"] = agent_event;

    // Thought events always populate delta.content with incremental tokens.
    // Answer events also populate delta.content when they carry text (e.g.
    // fallback full-content answer after tool_calls where the provider did
    // not stream deltas).
    if (ev.type == agent::AgentStreamEventType::Thought && !ev.text.empty())
    {
        choice["delta"]["content"] = ev.text;
    }
    else if (ev.type == agent::AgentStreamEventType::Answer && !ev.text.empty())
    {
        choice["delta"]["content"] = ev.text;
    }

    json j = {
        {"id",      "chatcmpl-lc-agent-stream"},
        {"object",  "chat.completion.chunk"},
        {"created", std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()},
        {"model",   model_id},
        {"choices", json::array({choice})}
    };
    return "data: " + j.dump() + "\n\n";
}

// --------- Query string parsing ---------
//
// Extracts key=value pairs from the query portion of a URL.
// Handles multiple values for the same key by keeping the last one.
std::unordered_map<std::string, std::string> parse_query(const std::string& query)
{
    std::unordered_map<std::string, std::string> result;
    std::size_t pos = 0;
    while (pos < query.size())
    {
        std::size_t amp = query.find('&', pos);
        std::string pair = query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        pos = amp == std::string::npos ? query.size() : amp + 1;

        std::size_t eq = pair.find('=');
        if (eq == std::string::npos)
        {
            continue;
        }
        std::string key = pair.substr(0, eq);
        std::string value = pair.substr(eq + 1);
        result[key] = value;
    }
    return result;
}

// --------- Path parameter matching ---------
//
// Tests whether a request path matches a route pattern that may contain
// :name placeholders. If it matches, fills path_params with the captured
// values and returns true. The query string (everything after '?') is
// ignored during matching.
//
// Examples:
//   match_path("/users/42", "/users/:id", params)  -- true, params["id"]="42"
//   match_path("/users/42/posts/7", "/users/:uid/posts/:pid", params)
//       -- true, params["uid"]="42", params["pid"]="7"
//   match_path("/users/42", "/users")              -- false
//   match_path("/users/42", "/users/:id/extra")    -- false
bool match_path(const std::string& req_path,
                const std::string& pattern,
                std::unordered_map<std::string, std::string>& path_params)
{
    path_params.clear();

    // Strip query string from request path for matching.
    std::size_t q = req_path.find('?');
    std::string clean_path = q == std::string::npos ? req_path : req_path.substr(0, q);

    std::vector<std::string> req_parts;
    std::vector<std::string> pat_parts;

    {
        std::istringstream iss(clean_path);
        std::string part;
        while (std::getline(iss, part, '/'))
        {
            if (!part.empty())
            {
                req_parts.push_back(part);
            }
        }
    }
    {
        std::istringstream iss(pattern);
        std::string part;
        while (std::getline(iss, part, '/'))
        {
            if (!part.empty())
            {
                pat_parts.push_back(part);
            }
        }
    }

    if (req_parts.size() != pat_parts.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < pat_parts.size(); ++i)
    {
        if (!pat_parts[i].empty() && pat_parts[i][0] == ':')
        {
            std::string name = pat_parts[i].substr(1);
            path_params[name] = req_parts[i];
        }
        else if (pat_parts[i] != req_parts[i])
        {
            return false;
        }
    }
    return true;
}

// --------- SSE framing helper ---------
//
// Wraps a raw payload string into a proper SSE frame:
//   data: <payload>\n\n
std::string sse_frame(const std::string& payload)
{
    return "data: " + payload + "\n\n";
}

std::string header_or(const httplib::Request& req, const std::string& name, const std::string& fallback = {})
{
    auto it = req.headers.find(name);
    return it == req.headers.end() ? fallback : it->second;
}

void log_api_request_start(const std::string& call_id,
                           const httplib::Request& req,
                           const std::string& route,
                           const std::string& model = {},
                           bool stream = false)
{
    std::ostringstream oss;
    oss << "[API_PERF][REQUEST] ApiServer perf start:"
        << " call_id=" << call_id
        << " method=" << req.method
        << " route=" << route
        << " path=" << req.path
        << " model=" << (model.empty() ? "-" : model)
        << " stream=" << (stream ? "true" : "false")
        << " req_bytes=" << req.body.size()
        << " session_id=" << header_or(req, "X-Session-Id", "-")
        << " remote_addr=" << (req.remote_addr.empty() ? "-" : req.remote_addr);
    LOG_INFO("{}", oss.str());
}

void log_api_response_end(const std::string& call_id,
                          const std::string& route,
                          int status,
                          std::chrono::steady_clock::time_point start,
                          std::size_t resp_bytes,
                          const std::string& model = {},
                          bool stream = false)
{
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::ostringstream oss;
    oss << "[API_PERF][RESPONSE] ApiServer perf end:"
        << " call_id=" << call_id
        << " route=" << route
        << " status=" << status
        << " elapsed_ms=" << (elapsed_us / 1000.0)
        << " elapsed_us=" << elapsed_us
        << " resp_bytes=" << resp_bytes
        << " model=" << (model.empty() ? "-" : model)
        << " stream=" << (stream ? "true" : "false");
    LOG_INFO("{}", oss.str());
}

} // namespace

// ---------------- ApiServer::Impl ----------------

struct ApiServer::Impl
{
    httplib::Server                 svr;
    std::mutex                      run_mu;
    std::condition_variable         run_cv;
    bool                            running = false;
};

ApiServer::ApiServer(ApiConfig cfg)
    : impl_(std::make_unique<Impl>()), cfg_(std::move(cfg))
{
}

ApiServer::~ApiServer()
{
    stop();
    if (worker_.joinable())
    {
        worker_.join();
    }
}

void ApiServer::register_model(std::string id, llm::LLMPtr llm)
{
    std::lock_guard<std::mutex> lk(mu_);
    models_[std::move(id)] = std::move(llm);
}

void ApiServer::register_agent(std::string id, std::shared_ptr<agent::ReActAgent> agent)
{
    std::lock_guard<std::mutex> lk(mu_);
    react_agents_[std::move(id)] = std::move(agent);
}

void ApiServer::register_agent(std::string id, std::shared_ptr<agent::ToolCallingAgent> agent)
{
    std::lock_guard<std::mutex> lk(mu_);
    tool_agents_[std::move(id)] = std::move(agent);
}

void ApiServer::set_hooks(hook::HookManager* mgr)
{
    std::lock_guard<std::mutex> lk(mu_);
    hooks_ = mgr;
    for (auto& kv : models_)
    {
        if (kv.second)
        {
            kv.second->set_hooks(mgr);
        }
    }
}

void ApiServer::add_route(const std::string& method,
                          const std::string& path,
                          RouteHandler handler)
{
    std::lock_guard<std::mutex> lk(mu_);
    routes_.push_back({method + " " + path, std::move(handler)});
}

void ApiServer::set_memory_resolver(MemoryResolver resolver)
{
    std::lock_guard<std::mutex> lk(mu_);
    memory_resolver_ = std::move(resolver);
}

bool ApiServer::is_running() const
{
    std::lock_guard<std::mutex> lk(impl_->run_mu);
    return impl_->running;
}

void ApiServer::stop()
{
    if (impl_)
    {
        // Only call svr.stop() when the server actually bound a socket.
        // On Windows/Debug, httplib asserts svr_sock_ != INVALID_SOCKET
        // inside stop(), which crashes if listen() never succeeded.
        {
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            if (!impl_->running) { return; }
        }
        impl_->svr.stop();
    }
    std::lock_guard<std::mutex> lk(impl_->run_mu);
    impl_->running = false;
    impl_->run_cv.notify_all();
}

void ApiServer::start()
{
    if (worker_.joinable())
    {
        bool running = false;
        {
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            running = impl_->running;
        }
        if (running)
        {
            throw LCError("ApiServer: already started");
        }
        worker_.join();
    }

    worker_ = std::thread([this]
    {
        try
        {
            run();
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("ApiServer worker stopped after exception: {}", e.what());
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            impl_->running = false;
            impl_->run_cv.notify_all();
        }
        catch (...)
        {
            LOG_ERROR("ApiServer worker stopped after unknown exception");
            std::lock_guard<std::mutex> lk(impl_->run_mu);
            impl_->running = false;
            impl_->run_cv.notify_all();
        }
    });

    // Block briefly until httplib reports the listener is up, so callers can
    // start firing requests immediately after start() returns.
    std::unique_lock<std::mutex> lk(impl_->run_mu);
    impl_->run_cv.wait_for(lk, std::chrono::seconds(5),
                           [this] { return impl_->running; });
}

void ApiServer::run()
{
    auto& svr = impl_->svr;

    auto auth_ok = [this](const httplib::Request& req) -> bool
    {
        if (cfg_.api_key.empty())
        {
            return true;
        }
        auto it = req.headers.find("Authorization");
        if (it == req.headers.end())
        {
            return false;
        }
        const std::string expected = "Bearer " + cfg_.api_key;
        return it->second == expected;
    };

    auto fire_api_event = [this](const std::string& route,
                                 const std::string& call_id,
                                 hook::Phase phase,
                                 std::chrono::microseconds elapsed = {},
                                 int status = 0)
    {
        auto* mgr = hooks_;
        if (!mgr)
        {
            return;
        }
        hook::HookContext ctx;
        ctx.phase     = phase;
        ctx.component = "ApiServer";
        ctx.call_id   = call_id;
        ctx.elapsed   = elapsed;
        ctx.metadata["route"] = route;
        if (status != 0)
        {
            ctx.metadata["status"] = status;
        }
        mgr->fire(ctx);
    };

    // ---- Helper: send a non-streaming Response back to httplib ----
    auto send_response = [](httplib::Response& hres, const Response& res)
    {
        hres.status = res.status;
        for (const auto& kv : res.headers)
        {
            hres.set_header(kv.first, kv.second);
        }
        hres.set_content(res.body, res.headers.count("Content-Type")
            ? res.headers.at("Content-Type")
            : "text/plain");
    };

    // ---- Helper: send a streaming Response back to httplib via SSE ----
    auto send_stream = [](httplib::Response& hres,
                          Response& res,
                          const std::function<void()>& on_complete)
    {
        hres.status = res.status;
        for (const auto& kv : res.headers)
        {
            hres.set_header(kv.first, kv.second);
        }
        hres.set_header("Cache-Control", "no-cache");
        hres.set_header("Connection", "keep-alive");

        // Capture the handler by value so it survives the lambda.
        auto handler = std::move(res.stream_handler);
        hres.set_chunked_content_provider(
            "text/event-stream",
            [handler, on_complete](std::size_t /*offset*/, httplib::DataSink& sink) -> bool
            {
                struct HttplibStreamSink : public StreamSink
                {
                    httplib::DataSink* sink_;
                    explicit HttplibStreamSink(httplib::DataSink* s) : sink_(s) {}
                    bool write(const std::string& data) override
                    {
                        std::string frame = sse_frame(data);
                        return sink_->write(frame.data(), frame.size());
                    }
                    void done() override
                    {
                        sink_->done();
                    }
                };

                HttplibStreamSink wrapper(&sink);
                try
                {
                    if (handler)
                    {
                        handler(wrapper);
                    }
                    else
                    {
                        wrapper.done();
                    }
                }
                catch (const std::exception& e)
                {
                    LOG_ERROR("ApiServer custom stream error: {}", e.what());
                    wrapper.write(json{{"error", e.what()}}.dump());
                    wrapper.done();
                }
                catch (...)
                {
                    LOG_ERROR("ApiServer custom stream unknown error");
                    wrapper.write(json{{"error", "unknown stream handler exception"}}.dump());
                    wrapper.done();
                }
                on_complete();
                return true;
            });
    };

    // ---- Custom route dispatch (reused for both pre_routing and generic handlers) ----
    auto dispatch_custom = [this, auth_ok, fire_api_event, send_response, send_stream]
        (const httplib::Request& hreq, httplib::Response& hres) -> bool
    {
        std::vector<std::pair<std::string, RouteHandler>> snapshot;
        {
            std::lock_guard<std::mutex> lk(mu_);
            snapshot = routes_;
        }

        for (const auto& entry : snapshot)
        {
            std::size_t sp = entry.first.find(' ');
            if (sp == std::string::npos)
            {
                continue;
            }
            std::string route_method = entry.first.substr(0, sp);
            std::string route_path   = entry.first.substr(sp + 1);

            if (route_method != hreq.method)
            {
                continue;
            }

            Request req;
            req.method = hreq.method;
            req.path   = hreq.path;
            req.body   = hreq.body;
            for (const auto& kv : hreq.headers)
            {
                req.headers[kv.first] = kv.second;
            }

            if (!match_path(hreq.path, route_path, req.path_params))
            {
                continue;
            }

            // Parse query string if present.
            std::size_t q = hreq.target.find('?');
            if (q != std::string::npos)
            {
                req.query_params = parse_query(hreq.target.substr(q + 1));
            }

            if (!cfg_.api_key.empty() && !auth_ok(hreq))
            {
                hres.status = 401;
                hres.set_content("{\"error\":\"unauthorized\"}", "application/json");
                return true;
            }

            std::string call_id = hooks_ ? hooks_->new_call_id() : "anon";
            auto start = std::chrono::steady_clock::now();
            log_api_request_start(call_id, hreq, route_path);
            fire_api_event(route_path, call_id, hook::Phase::BeforeApi);

            Response res;
            try
            {
                entry.second(req, res);
            }
            catch (const std::exception& e)
            {
                res.status = 500;
                json err = {{"error", e.what()}};
                res.set_json(err);
            }
            catch (...)
            {
                res.status = 500;
                json err = {{"error", "unknown route handler exception"}};
                res.set_json(err);
            }

            if (res.stream)
            {
                send_stream(hres, res,
                    [fire_api_event, route_path, call_id, start, status = res.status]()
                    {
                        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - start);
                        log_api_response_end(call_id, route_path, status, start, 0, {}, true);
                        fire_api_event(route_path, call_id, hook::Phase::AfterApi, elapsed);
                    });
            }
            else
            {
                send_response(hres, res);
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start);
                log_api_response_end(call_id, route_path, res.status, start, res.body.size());
                fire_api_event(route_path, call_id, hook::Phase::AfterApi, elapsed, res.status);
            }

            return true;
        }

        return false;
    };

    // For GET requests body is never needed; pre_routing_handler is fine.
    svr.set_pre_routing_handler(
        [dispatch_custom](const httplib::Request& hreq, httplib::Response& hres) ->
        httplib::Server::HandlerResponse
    {
        if (hreq.method != "GET")
        {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (dispatch_custom(hreq, hres))
        {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/healthz", [](const httplib::Request& req, httplib::Response& res)
    {
        std::string call_id = "healthz-" + std::to_string(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        auto start = std::chrono::steady_clock::now();
        log_api_request_start(call_id, req, "/healthz");
        constexpr const char* body = "{\"status\":\"ok\"}";
        res.set_content(body, "application/json");
        log_api_response_end(call_id, "/healthz", res.status, start, std::string(body).size());
    });

    svr.Get("/v1/models", [this, auth_ok](const httplib::Request& req, httplib::Response& res)
    {
        std::string call_id = "models-" + std::to_string(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        auto start = std::chrono::steady_clock::now();
        log_api_request_start(call_id, req, "/v1/models");
        if (!auth_ok(req))
        {
            res.status = 401;
            std::string body = "{\"error\":\"unauthorized\"}";
            res.set_content(body, "application/json");
            log_api_response_end(call_id, "/v1/models", res.status, start, body.size());
            return;
        }
        json arr = json::array();
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& kv : models_)
            {
                arr.push_back({
                    {"id",       kv.first},
                    {"object",   "model"},
                    {"owned_by", "langchain.cpp"}
                });
            }
        }
        json out = {{"object", "list"}, {"data", arr}};
        std::string body = out.dump();
        res.set_content(body, "application/json");
        log_api_response_end(call_id, "/v1/models", res.status, start, body.size());
    });

    svr.Post("/v1/chat/completions",
        [this, auth_ok, fire_api_event](const httplib::Request& req, httplib::Response& res)
    {
        std::string call_id = hooks_ ? hooks_->new_call_id() : "anon";
        auto start = std::chrono::steady_clock::now();

        if (!auth_ok(req))
        {
            res.status = 401;
            std::string body = "{\"error\":\"unauthorized\"}";
            res.set_content(body, "application/json");
            log_api_request_start(call_id, req, "/v1/chat/completions");
            log_api_response_end(call_id, "/v1/chat/completions", res.status, start, body.size());
            return;
        }

        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded())
        {
            res.status = 400;
            std::string err_body = "{\"error\":\"invalid JSON\"}";
            res.set_content(err_body, "application/json");
            log_api_request_start(call_id, req, "/v1/chat/completions");
            log_api_response_end(call_id, "/v1/chat/completions", res.status, start, err_body.size());
            return;
        }

        std::string model_id = body.value("model", std::string());

        // Resolve backend: agent takes priority over raw LLM.
        llm::LLMPtr backend;
        std::shared_ptr<agent::ReActAgent>       react_agent;
        std::shared_ptr<agent::ToolCallingAgent> tool_agent;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto ra_it = react_agents_.find(model_id);
            if (ra_it != react_agents_.end())
            {
                react_agent = ra_it->second;
            }
            else
            {
                auto ta_it = tool_agents_.find(model_id);
                if (ta_it != tool_agents_.end())
                {
                    tool_agent = ta_it->second;
                }
            }

            if (!react_agent && !tool_agent)
            {
                auto it = models_.find(model_id);
                if (it != models_.end())
                {
                    backend = it->second;
                }
                else if (!models_.empty())
                {
                    backend = models_.begin()->second;
                    if (model_id.empty())
                    {
                        model_id = models_.begin()->first;
                    }
                }
            }
        }
        if (!backend && !react_agent && !tool_agent)
        {
            res.status = 404;
            std::string err_body = "{\"error\":\"no model registered\"}";
            res.set_content(err_body, "application/json");
            log_api_request_start(call_id, req, "/v1/chat/completions", model_id);
            log_api_response_end(call_id, "/v1/chat/completions", res.status, start, err_body.size(), model_id);
            return;
        }

        bool stream_hint = body.value("stream", false);
        log_api_request_start(call_id, req, "/v1/chat/completions", model_id, stream_hint);
        fire_api_event("/v1/chat/completions", call_id, hook::Phase::BeforeLLM);

        llm::ChatRequest chat_req;
        try
        {
            chat_req = decode_request(body);
        }
        catch (const std::exception& e)
        {
            LOG_WARN("ApiServer bad request: call_id={} error={}", call_id, e.what());
            res.status = 400;
            json err = {{"error", std::string("invalid request: ") + e.what()}};
            std::string err_body = err.dump();
            res.set_content(err_body, "application/json");
            log_api_response_end(call_id, "/v1/chat/completions", res.status, start, err_body.size(), model_id);
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start);
            fire_api_event("/v1/chat/completions", call_id, hook::Phase::AfterLLM, elapsed, res.status);
            return;
        }
        catch (...)
        {
            LOG_WARN("ApiServer bad request: call_id={} unknown decode error", call_id);
            res.status = 400;
            json err = {{"error", "invalid request"}};
            std::string err_body = err.dump();
            res.set_content(err_body, "application/json");
            log_api_response_end(call_id, "/v1/chat/completions", res.status, start, err_body.size(), model_id);
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start);
            fire_api_event("/v1/chat/completions", call_id, hook::Phase::AfterLLM, elapsed, res.status);
            return;
        }
        bool is_stream = chat_req.stream;

        LOG_INFO("ApiServer request: call_id={} path=/v1/chat/completions model={} stream={}",
                 call_id, model_id, is_stream);

        // Extract the last user message for agent-based routing. We keep the
        // full Message (including multimodal content_parts) so that images
        // carried as base64 reach the agent/LLM and get persisted to memory.
        // user_prompt is also kept as a string projection for legacy code
        // paths and logging.
        std::string user_prompt;
        Message     user_message;
        for (const auto& m : chat_req.messages)
        {
            if (m.role == Role::User)
            {
                user_prompt  = m.content;
                user_message = m;
            }
        }

        // Resolve per-request memory (e.g. by session id) so a single shared
        // Agent can serve many sessions without leaking state between them.
        // The application controls isolation policy via the memory resolver.
        memory::MemoryPtr per_request_mem;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (memory_resolver_)
            {
                // Build a framework Request mirror for the resolver.
                Request r;
                r.method = req.method;
                r.path   = req.path;
                r.body   = req.body;
                for (const auto& kv : req.headers)
                {
                    r.headers[kv.first] = kv.second;
                }
                try
                {
                    per_request_mem = memory_resolver_(r);
                }
                catch (const std::exception& e)
                {
                    LOG_WARN("memory_resolver threw: {}", e.what());
                }
            }
        }

        // ---- Agent path ----
        if (react_agent || tool_agent)
        {
            if (is_stream)
            {
                Response framework_res;
                framework_res.status = 200;
                framework_res.enable_streaming(
                    [react_agent, tool_agent, user_message, chat_req, model_id, call_id, start, fire_api_event, per_request_mem]
                    (StreamSink& sink)
                {
                    try
                    {
                        // Use the rich AgentStreamCallback so the client sees
                        // every step: thought, tool_call, observation, answer.
                        auto on_event = [&](const agent::AgentStreamEvent& ev) -> bool
                        {
                            LOG_DEBUG("ApiServer agent stream: call_id={} type={}", call_id,
                                [&]() -> std::string {
                                    switch (ev.type)
                                    {
                                        case agent::AgentStreamEventType::Thought:     return "thought";
                                        case agent::AgentStreamEventType::ToolCall:    return "tool_call";
                                        case agent::AgentStreamEventType::Observation: return "observation";
                                        case agent::AgentStreamEventType::Answer:      return "answer";
                                        case agent::AgentStreamEventType::Error:       return "error";
                                        case agent::AgentStreamEventType::Done:        return "done";
                                    }
                                    return "unknown";
                                }());
                            return sink.write(make_agent_sse_chunk(model_id, ev, ""));
                        };

                        agent::AgentResult agent_result;
                        if (react_agent)
                        {
                            agent_result = react_agent->invoke_stream(
                                user_message, on_event,
                                chat_req.temperature,
                                chat_req.max_tokens,
                                chat_req.top_p,
                                chat_req.top_k,
                                per_request_mem);
                        }
                        else
                        {
                            agent_result = tool_agent->invoke_stream(
                                user_message, on_event,
                                chat_req.temperature,
                                chat_req.max_tokens,
                                chat_req.top_p,
                                chat_req.top_k,
                                per_request_mem);
                        }

                        // Final stop chunk for OpenAI compatibility
                        sink.write(make_sse_chunk(model_id, "", "stop"));
                        sink.write(sse_frame("[DONE]"));
                        LOG_INFO("ApiServer stream complete: call_id={} output_len={}",
                                 call_id, agent_result.output.size());
                    }
                    catch (const std::exception& e)
                    {
                        LOG_ERROR("ApiServer stream error: call_id={} error={}", call_id, e.what());
                        json err = {{"error", e.what()}};
                        sink.write(sse_frame(err.dump()));
                    }
                    sink.done();

                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start);
                    log_api_response_end(call_id, "/v1/chat/completions", 200, start, 0, model_id, true);
                    fire_api_event("/v1/chat/completions", call_id, hook::Phase::AfterLLM, elapsed);
                });

                res.status = framework_res.status;
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");

                auto handler = std::move(framework_res.stream_handler);
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [handler](std::size_t /*offset*/, httplib::DataSink& sink) -> bool
                    {
                        struct HttplibStreamSink : public StreamSink
                        {
                            httplib::DataSink* sink_;
                            explicit HttplibStreamSink(httplib::DataSink* s) : sink_(s) {}
                            bool write(const std::string& data) override
                            {
                                return sink_->write(data.data(), data.size());
                            }
                            void done() override
                            {
                                sink_->done();
                            }
                        };

                        HttplibStreamSink wrapper(&sink);
                        try
                        {
                            if (handler)
                            {
                                handler(wrapper);
                            }
                            else
                            {
                                wrapper.done();
                            }
                        }
                        catch (const std::exception& e)
                        {
                            LOG_ERROR("ApiServer stream provider error: {}", e.what());
                            json err = {{"error", e.what()}};
                            wrapper.write(sse_frame(err.dump()));
                            wrapper.done();
                        }
                        catch (...)
                        {
                            LOG_ERROR("ApiServer stream provider unknown error");
                            json err = {{"error", "unknown stream provider exception"}};
                            wrapper.write(sse_frame(err.dump()));
                            wrapper.done();
                        }
                        return true;
                    });
            }
            else
            {
                try
                {
                    agent::AgentResult agent_result;
                    if (react_agent)
                    {
                        agent_result = react_agent->invoke(
                            user_message,
                            chat_req.temperature,
                            chat_req.max_tokens,
                            chat_req.top_p,
                            chat_req.top_k,
                            per_request_mem);
                    }
                    else
                    {
                        agent_result = tool_agent->invoke(
                            user_message,
                            chat_req.temperature,
                            chat_req.max_tokens,
                            chat_req.top_p,
                            chat_req.top_k,
                            per_request_mem);
                    }

                    llm::ChatResponse fake_resp;
                    fake_resp.message = Message::assistant(agent_result.output);
                    fake_resp.finish_reason = agent_result.finished ? "stop" : "length";
                    auto out = encode_response(fake_resp, model_id);
                    res.set_content(out.dump(), "application/json");
                    LOG_INFO("ApiServer response: call_id={} status=200 output_len={}",
                             call_id, agent_result.output.size());
                }
                catch (const std::exception& e)
                {
                    LOG_ERROR("ApiServer error: call_id={} error={}", call_id, e.what());
                    res.status = 500;
                    json err = {{"error", e.what()}};
                    res.set_content(err.dump(), "application/json");
                }

                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start);
                log_api_response_end(call_id, "/v1/chat/completions", res.status, start,
                                    res.body.size(), model_id, false);
                fire_api_event("/v1/chat/completions", call_id, hook::Phase::AfterLLM, elapsed);
            }
            return;
        }

        // ---- Raw LLM path ----
        if (is_stream)
        {
            Response framework_res;
            framework_res.status = 200;
            framework_res.enable_streaming(
                [backend, chat_req, model_id, call_id, start, fire_api_event]
                (StreamSink& sink)
            {
                try
                {
                    backend->invoke_stream(chat_req,
                        [&](const std::string& delta) -> bool
                        {
                            LOG_DEBUG("ApiServer stream chunk: call_id={} delta_len={}", call_id, delta.size());
                            return sink.write(make_sse_chunk(model_id, delta, ""));
                        });

                    sink.write(make_sse_chunk(model_id, "", "stop"));
                    sink.write(sse_frame("[DONE]"));
                    LOG_INFO("ApiServer stream complete: call_id={}", call_id);
                }
                catch (const std::exception& e)
                {
                    LOG_ERROR("ApiServer stream error: call_id={} error={}", call_id, e.what());
                    json err = {{"error", e.what()}};
                    sink.write(sse_frame(err.dump()));
                }
                sink.done();

                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start);
                log_api_response_end(call_id, "/v1/chat/completions", 200, start, 0, model_id, true);
                fire_api_event("/v1/chat/completions", call_id,
                               hook::Phase::AfterLLM, elapsed);
            });

            res.status = framework_res.status;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            auto handler = std::move(framework_res.stream_handler);
            res.set_chunked_content_provider(
                "text/event-stream",
                [handler](std::size_t /*offset*/, httplib::DataSink& sink) -> bool
                {
                    struct HttplibStreamSink : public StreamSink
                    {
                        httplib::DataSink* sink_;
                        explicit HttplibStreamSink(httplib::DataSink* s) : sink_(s) {}
                        bool write(const std::string& data) override
                        {
                            return sink_->write(data.data(), data.size());
                        }
                        void done() override
                        {
                            sink_->done();
                        }
                    };

                    HttplibStreamSink wrapper(&sink);
                    try
                    {
                        if (handler)
                        {
                            handler(wrapper);
                        }
                        else
                        {
                            wrapper.done();
                        }
                    }
                    catch (const std::exception& e)
                    {
                        LOG_ERROR("ApiServer stream provider error: {}", e.what());
                        json err = {{"error", e.what()}};
                        wrapper.write(sse_frame(err.dump()));
                        wrapper.done();
                    }
                    catch (...)
                    {
                        LOG_ERROR("ApiServer stream provider unknown error");
                        json err = {{"error", "unknown stream provider exception"}};
                        wrapper.write(sse_frame(err.dump()));
                        wrapper.done();
                    }
                    return true;
                });
        }
        else
        {
            try
            {
                auto resp = backend->invoke(chat_req);
                auto out  = encode_response(resp, model_id);
                res.set_content(out.dump(), "application/json");
                LOG_INFO("ApiServer response: call_id={} status=200 content_len={}",
                         call_id, resp.message.content.size());
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("ApiServer error: call_id={} error={}", call_id, e.what());
                res.status = 500;
                json err = {{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }

            log_api_response_end(call_id, "/v1/chat/completions", res.status, start,
                                res.body.size(), model_id, false);
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start);
            fire_api_event("/v1/chat/completions", call_id, hook::Phase::AfterLLM, elapsed);
        }
    });

    svr.set_read_timeout(cfg_.read_timeout_sec, 0);
    svr.set_write_timeout(cfg_.write_timeout_sec, 0);

    // Generic handlers for custom routes with bodies (POST/PUT/PATCH/DELETE).
    // These run after the explicit route table, so built-in routes take precedence.
    auto generic_custom_handler = [dispatch_custom](const httplib::Request& hreq,
                                                     httplib::Response& hres)
    {
        dispatch_custom(hreq, hres);
    };

    svr.Post(".*", generic_custom_handler);
    svr.Put(".*", generic_custom_handler);
    svr.Patch(".*", generic_custom_handler);
    svr.Delete(".*", generic_custom_handler);

    if (!svr.bind_to_port(cfg_.host.c_str(), cfg_.port))
    {
        std::lock_guard<std::mutex> lk(impl_->run_mu);
        impl_->running = false;
        impl_->run_cv.notify_all();
        return;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->run_mu);
        impl_->running = true;
        impl_->run_cv.notify_all();
    }

    svr.listen_after_bind();

    {
        std::lock_guard<std::mutex> lk(impl_->run_mu);
        impl_->running = false;
    }
}

} // namespace api
} // namespace langchain
