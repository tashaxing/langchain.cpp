// src/api/client.cpp -- generic HTTP client for AI services.
#include "api/client.h"

#include "util/strings.h"

#include <httplib.h>

#include <sstream>

namespace langchain
{
namespace api
{

namespace
{

// Split "https://host:port" into the form cpp-httplib expects.
struct ParsedUrl
{
    std::string scheme_host;
    std::string path_prefix;
};

ParsedUrl split_base_url(const std::string& url)
{
    ParsedUrl out;
    auto pos = url.find("://");
    if (pos == std::string::npos)
    {
        out.scheme_host = url;
        return out;
    }
    auto path_pos = url.find('/', pos + 3);
    if (path_pos == std::string::npos)
    {
        out.scheme_host = url;
        return out;
    }
    out.scheme_host = url.substr(0, path_pos);
    out.path_prefix = url.substr(path_pos);
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------

json HttpResponse::json_body() const
{
    if (body.empty())
    {
        return json{};
    }
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded())
    {
        return json{};
    }
    return j;
}

// ---------------------------------------------------------------------------
// HttpClient::Impl
// ---------------------------------------------------------------------------

struct HttpClient::Impl
{
    mutable std::mutex mu;
    HttpClientConfig cfg;

    explicit Impl(HttpClientConfig c) : cfg(std::move(c)) {}

    httplib::Headers build_headers(
        const std::unordered_map<std::string, std::string>& extra) const
    {
        httplib::Headers h;
        std::lock_guard<std::mutex> lk(mu);
        if (!cfg.api_key.empty())
        {
            h.emplace("Authorization", "Bearer " + cfg.api_key);
        }
        for (const auto& kv : cfg.extra_headers)
        {
            h.emplace(kv.first, kv.second);
        }
        for (const auto& kv : extra)
        {
            h.emplace(kv.first, kv.second);
        }
        return h;
    }

    std::unique_ptr<httplib::Client> make_client(const ParsedUrl& parsed) const
    {
        auto cli = std::make_unique<httplib::Client>(parsed.scheme_host);
        std::lock_guard<std::mutex> lk(mu);
        cli->set_connection_timeout(cfg.connect_timeout_sec);
        cli->set_read_timeout(cfg.read_timeout_sec);
        return cli;
    }
};

// ---------------------------------------------------------------------------
// HttpClient public
// ---------------------------------------------------------------------------

HttpClient::HttpClient(HttpClientConfig cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg)))
{
}

HttpClient::~HttpClient() = default;

HttpClientConfig HttpClient::config() const
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->cfg;
}

void HttpClient::set_config(const HttpClientConfig& cfg)
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->cfg = cfg;
}

HttpResponse HttpClient::request(const std::string& method,
                                  const std::string& path,
                                  const std::string& body,
                                  const std::unordered_map<std::string, std::string>& extra_headers) const
{
    auto parsed = split_base_url(impl_->cfg.base_url);
    auto cli = impl_->make_client(parsed);
    auto headers = impl_->build_headers(extra_headers);
    auto target = parsed.path_prefix + path;

    std::unique_ptr<httplib::Result> res;
    if (method == "GET")
    {
        res = std::make_unique<httplib::Result>(cli->Get(target.c_str(), headers));
    }
    else if (method == "POST")
    {
        res = std::make_unique<httplib::Result>(cli->Post(target.c_str(), headers, body, "application/json"));
    }
    else if (method == "PUT")
    {
        res = std::make_unique<httplib::Result>(cli->Put(target.c_str(), headers, body, "application/json"));
    }
    else if (method == "DELETE")
    {
        res = std::make_unique<httplib::Result>(cli->Delete(target.c_str(), headers));
    }
    else if (method == "PATCH")
    {
        res = std::make_unique<httplib::Result>(cli->Patch(target.c_str(), headers, body, "application/json"));
    }
    else
    {
        throw LCError("HttpClient: unsupported method: " + method);
    }

    HttpResponse out;
    if (!res || !*res)
    {
        out.status = 0;
        return out;
    }
    out.status = (*res)->status;
    out.body   = (*res)->body;
    for (const auto& kv : (*res)->headers)
    {
        out.headers[kv.first] = kv.second;
    }
    return out;
}

HttpResponse HttpClient::get(const std::string& path,
                              const std::unordered_map<std::string, std::string>& extra_headers) const
{
    return request("GET", path, {}, extra_headers);
}

HttpResponse HttpClient::post(const std::string& path,
                               const std::string& body,
                               const std::unordered_map<std::string, std::string>& extra_headers) const
{
    return request("POST", path, body, extra_headers);
}

HttpResponse HttpClient::put(const std::string& path,
                              const std::string& body,
                              const std::unordered_map<std::string, std::string>& extra_headers) const
{
    return request("PUT", path, body, extra_headers);
}

HttpResponse HttpClient::del(const std::string& path,
                              const std::unordered_map<std::string, std::string>& extra_headers) const
{
    return request("DELETE", path, {}, extra_headers);
}

HttpResponse HttpClient::json_get(const std::string& path,
                                   const std::unordered_map<std::string, std::string>& extra_headers) const
{
    auto h = extra_headers;
    h["Accept"] = "application/json";
    auto res = get(path, h);
    return res;
}

HttpResponse HttpClient::json_post(const std::string& path,
                                    const json& body,
                                    const std::unordered_map<std::string, std::string>& extra_headers) const
{
    auto h = extra_headers;
    h["Content-Type"] = "application/json";
    h["Accept"] = "application/json";
    return post(path, body.dump(), h);
}

HttpResponse HttpClient::json_put(const std::string& path,
                                   const json& body,
                                   const std::unordered_map<std::string, std::string>& extra_headers) const
{
    auto h = extra_headers;
    h["Content-Type"] = "application/json";
    h["Accept"] = "application/json";
    return put(path, body.dump(), h);
}

HttpResponse HttpClient::json_del(const std::string& path,
                                    const std::unordered_map<std::string, std::string>& extra_headers) const
{
    auto h = extra_headers;
    h["Accept"] = "application/json";
    return del(path, h);
}

HttpResponse HttpClient::stream_post(const std::string& path,
                                      const std::string& body,
                                      const std::function<bool(const std::string& line)>& on_line,
                                      const std::unordered_map<std::string, std::string>& extra_headers) const
{
    auto parsed = split_base_url(impl_->cfg.base_url);
    auto cli = impl_->make_client(parsed);
    auto headers = impl_->build_headers(extra_headers);
    headers.emplace("Accept", "text/event-stream");

    std::string buffer;
    bool aborted = false;
    auto target = parsed.path_prefix + path;

    auto res = cli->Post(
        target.c_str(),
        headers,
        body,
        "application/json",
        [&](const char* data, std::size_t len) -> bool
        {
            if (aborted)
            {
                return false;
            }
            buffer.append(data, len);
            // SSE frames are separated by blank lines.
            std::size_t pos;
            while ((pos = buffer.find("\n\n")) != std::string::npos)
            {
                std::string frame = buffer.substr(0, pos);
                buffer.erase(0, pos + 2);
                for (const auto& line : strings::split(frame, '\n'))
                {
                    auto trimmed = strings::trim(line);
                    if (!trimmed.empty() && !strings::starts_with(trimmed, "data:"))
                    {
                        continue;
                    }
                    if (on_line && !on_line(trimmed))
                    {
                        aborted = true;
                        return false;
                    }
                }
            }
            return true;
        });

    HttpResponse out;
    if (!res)
    {
        out.status = 0;
        return out;
    }
    out.status = res->status;
    out.body   = res->body;
    for (const auto& kv : res->headers)
    {
        out.headers[kv.first] = kv.second;
    }
    return out;
}

// ==========================================================================
// AIClient
// ==========================================================================

namespace
{

json content_part_to_json(const ContentPart& part)
{
    if (part.type == "text")
    {
        return json{{"type", "text"}, {"text", part.text}};
    }
    if (part.type == "image_url")
    {
        return json{{"type", "image_url"}, {"image_url", {{"url", part.url}}}};
    }
    if (part.type == "image_base64")
    {
        std::string data_url = "data:" + part.mime_type + ";base64," + part.base64_data;
        return json{{"type", "image_url"}, {"image_url", {{"url", data_url}}}};
    }
    return json{{"type", part.type}};
}

json message_to_json(const Message& m)
{
    json j;
    j["role"] = to_string(m.role);

    if (!m.content_parts.empty())
    {
        json arr = json::array();
        for (const auto& part : m.content_parts)
        {
            arr.push_back(content_part_to_json(part));
        }
        j["content"] = std::move(arr);
    }
    else if (!m.content.empty())
    {
        j["content"] = m.content;
    }

    if (!m.name.empty())
    {
        j["name"] = m.name;
    }
    if (!m.tool_call_id.empty())
    {
        j["tool_call_id"] = m.tool_call_id;
    }
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

Message json_to_message(const json& j)
{
    Message m;
    m.role = role_from_string(j.value("role", std::string("assistant")));

    if (j.contains("content") && j["content"].is_array())
    {
        for (const auto& part : j["content"])
        {
            std::string type = part.value("type", std::string("text"));
            if (type == "text")
            {
                m.content_parts.push_back(ContentPart::text_part(
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
                        m.content_parts.push_back(ContentPart::image_base64(std::move(data), mime));
                    }
                    else
                    {
                        m.content_parts.push_back(ContentPart::image_url(std::move(url)));
                    }
                }
                else
                {
                    m.content_parts.push_back(ContentPart::image_url(std::move(url)));
                }
            }
            else
            {
                m.content_parts.push_back(ContentPart::text_part(part.dump()));
            }
        }
        if (!m.content_parts.empty())
        {
            std::ostringstream oss;
            bool first = true;
            for (const auto& p : m.content_parts)
            {
                if (!first) oss << "\n";
                first = false;
                if (p.type == "text") oss << p.text;
                else if (p.type == "image_url") oss << "[image_url: " << p.url << "]";
                else if (p.type == "image_base64") oss << "[image_base64: " << p.mime_type << "]";
            }
            m.content = oss.str();
        }
    }
    else if (j.contains("content") && j["content"].is_string())
    {
        m.content = j["content"].get<std::string>();
    }

    if (j.contains("tool_calls") && j["tool_calls"].is_array())
    {
        for (const auto& tc : j["tool_calls"])
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
    return m;
}

json build_chat_payload(const llm::ChatRequest& req, const std::string& model, bool stream)
{
    json payload;
    payload["model"]  = model;
    payload["stream"] = stream;
    json msgs = json::array();
    for (const auto& m : req.messages)
    {
        msgs.push_back(message_to_json(m));
    }
    payload["messages"] = std::move(msgs);

    if (req.temperature)
    {
        payload["temperature"] = *req.temperature;
    }
    if (req.max_tokens)
    {
        payload["max_tokens"] = *req.max_tokens;
    }
    if (req.top_p)
    {
        payload["top_p"] = *req.top_p;
    }
    if (req.top_k)
    {
        payload["top_k"] = *req.top_k;
    }
    if (!req.stop.empty())
    {
        payload["stop"] = req.stop;
    }
    if (!req.tools.empty())
    {
        json tools = json::array();
        for (const auto& t : req.tools)
        {
            tools.push_back({
                {"type", "function"},
                {"function", {
                    {"name", t.name},
                    {"description", t.description},
                    {"parameters", t.parameters}
                }}
            });
        }
        payload["tools"] = std::move(tools);
        if (req.tool_choice)
        {
            payload["tool_choice"] = *req.tool_choice;
        }
    }
    return payload;
}

} // namespace

// ---------------- AIClient::Impl ----------------

struct AIClient::Impl
{
    HttpClient http;
    mutable std::mutex mu;
    HttpClientConfig cfg;

    explicit Impl(HttpClientConfig c)
        : http(std::move(c)),
          cfg(http.config())
    {
    }

    std::string effective_model(const std::string& model_override) const
    {
        std::lock_guard<std::mutex> lk(mu);
        return model_override.empty() ? cfg.model : model_override;
    }
};

AIClient::AIClient(HttpClientConfig cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg)))
{
}

AIClient::~AIClient() = default;

HttpClient& AIClient::http()
{
    return impl_->http;
}

const HttpClient& AIClient::http() const
{
    return impl_->http;
}

HttpClientConfig AIClient::config() const
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->cfg;
}

void AIClient::set_config(const HttpClientConfig& cfg)
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->cfg = cfg;
    impl_->http.set_config(cfg);
}

// ---------------------------------------------------------------------------
// Chat (non-streaming)
// ---------------------------------------------------------------------------

llm::ChatResponse AIClient::invoke(const llm::ChatRequest& req, const std::string& model)
{
    std::string m = impl_->effective_model(model);
    auto body = build_chat_payload(req, m, false);
    auto res = impl_->http.json_post("/v1/chat/completions", body);
    if (!res.ok())
    {
        throw LCError("AIClient: invoke failed HTTP " + std::to_string(res.status) + ": " + res.body);
    }

    json j = res.json_body();
    llm::ChatResponse out;
    out.model = j.value("model", m);
    const auto& choices = j.value("choices", json::array());
    if (!choices.empty())
    {
        const auto& c0 = choices[0];
        out.message = json_to_message(c0.value("message", json::object()));
        out.finish_reason = c0.value("finish_reason", std::string());
    }
    if (j.contains("usage"))
    {
        const auto& u = j["usage"];
        out.usage.prompt_tokens     = u.value("prompt_tokens", 0);
        out.usage.completion_tokens = u.value("completion_tokens", 0);
        out.usage.total_tokens      = u.value("total_tokens", 0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Chat (streaming / SSE)
// ---------------------------------------------------------------------------

llm::ChatResponse AIClient::invoke_stream(const llm::ChatRequest& req,
                                            const StreamChunkCallback& on_delta,
                                            const std::string& model)
{
    std::string m = impl_->effective_model(model);
    auto body = build_chat_payload(req, m, true).dump();

    std::string aggregate_content;
    std::string finish_reason;

    auto res = impl_->http.stream_post(
        "/v1/chat/completions",
        body,
        [&](const std::string& line) -> bool
        {
            if (!strings::starts_with(line, "data:"))
            {
                return true;
            }
            std::string payload = strings::trim(line.substr(5));
            if (payload == "[DONE]")
            {
                return true;
            }
            json j = json::parse(payload, nullptr, false);
            if (j.is_discarded())
            {
                return true;
            }
            const auto& choices = j.value("choices", json::array());
            if (choices.empty())
            {
                return true;
            }
            const auto& c0 = choices[0];
            if (c0.contains("delta"))
            {
                const auto& d = c0["delta"];
                if (d.contains("content") && d["content"].is_string())
                {
                    std::string piece = d["content"].get<std::string>();
                    aggregate_content += piece;
                    if (on_delta && !on_delta(piece))
                    {
                        return false;
                    }
                }
            }
            if (c0.contains("finish_reason") && c0["finish_reason"].is_string())
            {
                finish_reason = c0["finish_reason"].get<std::string>();
            }
            return true;
        });

    if (!res.ok())
    {
        throw LCError("AIClient: invoke_stream failed HTTP " + std::to_string(res.status) + ": " + res.body);
    }

    llm::ChatResponse out;
    out.model = m;
    out.message = Message::assistant(aggregate_content);
    out.finish_reason = finish_reason.empty() ? std::string("stop") : finish_reason;
    return out;
}

// ---------------------------------------------------------------------------
// Convenience
// ---------------------------------------------------------------------------

std::string AIClient::complete(const std::string& prompt, const std::string& model)
{
    llm::ChatRequest req;
    req.messages.push_back(Message::user(prompt));
    return invoke(req, model).message.content;
}

std::string AIClient::complete_stream(const std::string& prompt,
                                        const StreamChunkCallback& on_delta,
                                        const std::string& model)
{
    llm::ChatRequest req;
    req.messages.push_back(Message::user(prompt));
    return invoke_stream(req, on_delta, model).message.content;
}

} // namespace api
} // namespace langchain
