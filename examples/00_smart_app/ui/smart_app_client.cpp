// smart_app_client.cpp -- HTTP client wrapper for 00_smart_app chat UI.
#include "smart_app_client.h"

#include "api/client.h"
#include "util/common.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>

namespace smart_app::ui
{

namespace
{

langchain::api::HttpClientConfig to_http_config(const ChatConfig& cfg)
{
    langchain::api::HttpClientConfig http_cfg;
    http_cfg.base_url = cfg.base_url();
    http_cfg.model = cfg.model;
    http_cfg.connect_timeout_sec = cfg.connect_timeout_sec;
    http_cfg.read_timeout_sec = cfg.read_timeout_sec;
    return http_cfg;
}

std::unordered_map<std::string, std::string> session_headers(const ChatConfig& cfg)
{
    std::unordered_map<std::string, std::string> headers;
    if (!cfg.session_id.empty())
    {
        headers["X-Session-Id"] = cfg.session_id;
    }
    return headers;
}

std::string response_error(const langchain::api::HttpResponse& res)
{
    std::ostringstream oss;
    if (res.status == 0)
    {
        oss << "network error";
    }
    else
    {
        oss << "HTTP " << res.status;
    }
    if (!res.body.empty())
    {
        oss << ": " << res.body;
    }
    return oss.str();
}

langchain::json make_chat_body(const ChatConfig& cfg, const std::vector<ChatMessage>& messages, bool stream)
{
    langchain::json body;
    body["model"] = cfg.model;
    body["stream"] = stream;
    body["temperature"] = cfg.temperature;
    body["top_p"] = cfg.top_p;
    body["top_k"] = cfg.top_k;
    body["max_tokens"] = cfg.max_tokens;

    langchain::json arr = langchain::json::array();
    for (const auto& m : messages)
    {
        arr.push_back({{"role", m.role}, {"content", m.content}});
    }
    body["messages"] = std::move(arr);
    return body;
}

std::string trim_sse_payload(std::string line)
{
    if (line.rfind("data:", 0) == 0)
    {
        line = line.substr(5);
    }
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
    {
        line.erase(line.begin());
    }
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
    {
        line.pop_back();
    }
    return line;
}

ChatEvent parse_sse_event(const std::string& payload)
{
    ChatEvent ev;
    ev.raw_json = payload;

    if (payload == "[DONE]")
    {
        ev.type = ChatEvent::Type::Done;
        return ev;
    }

    auto j = langchain::json::parse(payload, nullptr, false);
    if (j.is_discarded())
    {
        ev.type = ChatEvent::Type::Error;
        ev.text = "invalid SSE JSON: " + payload;
        return ev;
    }

    if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty())
    {
        return ev;
    }

    const auto& choice = j["choices"][0];
    if (choice.contains("agent_event") && choice["agent_event"].is_object())
    {
        const auto& ae = choice["agent_event"];
        ev.type = ChatEvent::Type::AgentEvent;
        ev.agent_event_type = ae.value("type", std::string());
        ev.text = ae.value("text", std::string());
        ev.tool_name = ae.value("tool_name", std::string());
        ev.tool_input = ae.value("tool_input", std::string());
        return ev;
    }

    if (choice.contains("delta") && choice["delta"].is_object())
    {
        const auto& delta = choice["delta"];
        if (delta.contains("content") && delta["content"].is_string())
        {
            ev.type = ChatEvent::Type::Delta;
            ev.text = delta["content"].get<std::string>();
            return ev;
        }
    }

    if (choice.contains("finish_reason") && !choice["finish_reason"].is_null())
    {
        ev.type = ChatEvent::Type::Done;
    }

    return ev;
}

} // namespace

SmartAppClient::SmartAppClient(ChatConfig config)
    : config_(std::move(config))
{
}

ChatConfig SmartAppClient::config() const
{
    return config_;
}

void SmartAppClient::set_config(ChatConfig config)
{
    config_ = std::move(config);
}

bool SmartAppClient::health(std::string* error) const
{
    langchain::api::HttpClient http(to_http_config(config_));
    auto res = http.json_get("/healthz");
    if (!res.ok())
    {
        if (error) *error = response_error(res);
        return false;
    }
    return true;
}

std::vector<ModelInfo> SmartAppClient::list_models(std::string* error) const
{
    langchain::api::HttpClient http(to_http_config(config_));
    auto res = http.json_get("/v1/models");
    if (!res.ok())
    {
        if (error) *error = response_error(res);
        return {};
    }

    std::vector<ModelInfo> out;
    auto j = res.json_body();
    if (j.contains("data") && j["data"].is_array())
    {
        for (const auto& item : j["data"])
        {
            std::string id = item.value("id", std::string());
            if (!id.empty())
            {
                out.push_back({id});
            }
        }
    }
    return out;
}

std::string SmartAppClient::chat_once(const std::vector<ChatMessage>& messages, std::string* error) const
{
    langchain::api::HttpClient http(to_http_config(config_));
    auto res = http.json_post("/v1/chat/completions", make_chat_body(config_, messages, false), session_headers(config_));
    if (!res.ok())
    {
        if (error) *error = response_error(res);
        return {};
    }

    auto j = res.json_body();
    if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty())
    {
        const auto& msg = j["choices"][0].value("message", langchain::json::object());
        return msg.value("content", std::string());
    }

    if (error) *error = "invalid chat response: " + res.body;
    return {};
}

bool SmartAppClient::chat_stream(const std::vector<ChatMessage>& messages,
                                 const std::function<bool(const ChatEvent&)>& on_event,
                                 std::string* error) const
{
    langchain::api::HttpClient http(to_http_config(config_));
    auto body = make_chat_body(config_, messages, true).dump();
    auto headers = session_headers(config_);
    headers["Content-Type"] = "application/json";

    auto res = http.stream_post("/v1/chat/completions", body,
        [&](const std::string& line) -> bool
    {
        auto payload = trim_sse_payload(line);
        if (payload.empty())
        {
            return true;
        }
        return on_event(parse_sse_event(payload));
    }, headers);

    if (!res.ok())
    {
        if (error) *error = response_error(res);
        return false;
    }
    return true;
}

bool SmartAppClient::clear_memory(std::string* error) const
{
    langchain::api::HttpClient http(to_http_config(config_));
    auto res = http.json_post("/v1/memory/clear", langchain::json::object(), session_headers(config_));
    if (!res.ok())
    {
        if (error) *error = response_error(res);
        return false;
    }
    return true;
}

} // namespace smart_app::ui
