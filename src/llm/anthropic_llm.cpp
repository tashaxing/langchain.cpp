// src/llm/anthropic_llm.cpp -- Anthropic Claude API backend.
#include "llm/anthropic_llm.h"

#include "util/strings.h"

#include <httplib.h>

#include <sstream>

namespace langchain
{
namespace llm
{

namespace
{

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

// Anthropic uses a different message format than OpenAI
json anthropic_message_to_json(const Message& m)
{
    json j;
    j["role"] = to_string(m.role);

    // Anthropic expects content as an array of blocks
    if (!m.content_parts.empty())
    {
        json arr = json::array();
        for (const auto& part : m.content_parts)
        {
            if (part.type == "text")
            {
                arr.push_back({{"type", "text"}, {"text", part.text}});
            }
            else if (part.type == "image_url")
            {
                // Convert to Anthropic's image format
                arr.push_back({{"type", "image"},
                    {"source", {{"type", "url"}, {"url", part.url}}}});
            }
            else if (part.type == "image_base64")
            {
                arr.push_back({{"type", "image"},
                    {"source", {{"type", "base64"},
                                {"media_type", part.mime_type},
                                {"data", part.base64_data}}}});
            }
        }
        j["content"] = std::move(arr);
    }
    else if (!m.content.empty())
    {
        // Simple text content as single block
        j["content"] = json::array({{{"type", "text"}, {"text", m.content}}});
    }

    return j;
}

Message anthropic_json_to_message(const json& j)
{
    Message m;
    m.role = role_from_string(j.value("role", std::string("assistant")));

    // Anthropic returns content as an array of blocks
    if (j.contains("content") && j["content"].is_array())
    {
        std::ostringstream oss;
        bool first = true;
        for (const auto& block : j["content"])
        {
            std::string type = block.value("type", std::string("text"));
            if (type == "text")
            {
                std::string text = block.value("text", std::string());
                m.content_parts.push_back(ContentPart::text_part(text));
                if (!first) oss << "\n";
                first = false;
                oss << text;
            }
            else if (type == "image")
            {
                const auto& source = block.value("source", json::object());
                std::string source_type = source.value("type", std::string());
                if (source_type == "url")
                {
                    std::string url = source.value("url", std::string());
                    m.content_parts.push_back(ContentPart::image_url(std::move(url)));
                }
                else if (source_type == "base64")
                {
                    std::string mime = source.value("media_type", "image/png");
                    std::string data = source.value("data", std::string());
                    m.content_parts.push_back(ContentPart::image_base64(std::move(data), mime));
                }
                if (!first) oss << "\n";
                first = false;
                oss << "[image]";
            }
        }
        m.content = oss.str();
    }
    else if (j.contains("content") && j["content"].is_string())
    {
        m.content = j["content"].get<std::string>();
    }

    return m;
}

} // namespace

AnthropicLLM::AnthropicLLM(AnthropicLLMConfig cfg)
    : cfg_(std::move(cfg))
{
}

json AnthropicLLM::build_payload(const ChatRequest& req, bool stream) const
{
    json payload;
    payload["model"] = cfg_.model;
    payload["stream"] = stream;
    payload["max_tokens"] = cfg_.max_tokens;

    // Anthropic uses "messages" array
    json msgs = json::array();
    for (const auto& m : req.messages)
    {
        msgs.push_back(anthropic_message_to_json(m));
    }
    payload["messages"] = std::move(msgs);

    if (req.temperature)
    {
        payload["temperature"] = *req.temperature;
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
        // Anthropic uses "stop_sequences"
        payload["stop_sequences"] = req.stop;
    }

    return payload;
}

Message AnthropicLLM::decode_response(const json& j) const
{
    return anthropic_json_to_message(j);
}

ChatResponse AnthropicLLM::invoke_impl(const ChatRequest& req)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{
        {"Content-Type", "application/json"},
        {"x-api-key", cfg_.api_key},
        {"anthropic-version", "2023-06-01"}
    };

    auto body = build_payload(req, false).dump();
    auto res = cli.Post(
        (parsed.path_prefix + "/v1/messages").c_str(),
        headers,
        body,
        "application/json");

    if (!res)
    {
        throw LCError("Anthropic: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status / 100 != 2)
    {
        throw LCError("Anthropic: HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("Anthropic: invalid JSON response");
    }

    ChatResponse out;
    out.model = j.value("model", cfg_.model);
    out.message = decode_response(j);
    out.finish_reason = j.value("stop_reason", std::string());
    if (out.finish_reason == "end_turn")
    {
        out.finish_reason = "stop";
    }

    if (j.contains("usage"))
    {
        const auto& u = j["usage"];
        out.usage.prompt_tokens     = u.value("input_tokens", 0);
        out.usage.completion_tokens = u.value("output_tokens", 0);
    }
    return out;
}

ChatResponse AnthropicLLM::invoke_stream_impl(const ChatRequest& req,
                                               const StreamCallback& on_delta)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{
        {"Content-Type", "application/json"},
        {"Accept", "text/event-stream"},
        {"x-api-key", cfg_.api_key},
        {"anthropic-version", "2023-06-01"}
    };

    auto body = build_payload(req, true).dump();

    std::string aggregate_content;
    std::string finish_reason;
    std::string buffer;
    bool aborted = false;

    auto handle_event = [&](const std::string& payload)
    {
        json j = json::parse(payload, nullptr, false);
        if (j.is_discarded())
        {
            return;
        }

        std::string type = j.value("type", std::string());
        if (type == "content_block_delta")
        {
            const auto& delta = j.value("delta", json::object());
            if (delta.contains("text"))
            {
                std::string piece = delta.value("text", std::string());
                aggregate_content += piece;
                if (!on_delta(piece))
                {
                    aborted = true;
                }
            }
        }
        else if (type == "message_stop")
        {
            const auto& msg = j.value("message", json::object());
            finish_reason = msg.value("stop_reason", std::string("stop"));
        }
        else if (type == "message_delta")
        {
            const auto& delta = j.value("delta", json::object());
            if (delta.contains("stop_reason"))
            {
                finish_reason = delta.value("stop_reason", std::string());
            }
        }
    };

    auto res = cli.Post(
        (parsed.path_prefix + "/v1/messages").c_str(),
        headers,
        body,
        "application/json",
        [&](const char* data, std::size_t len)
        {
            if (aborted)
            {
                return false;
            }
            buffer.append(data, len);
            std::size_t pos;
            while ((pos = buffer.find("\n\n")) != std::string::npos)
            {
                std::string frame = buffer.substr(0, pos);
                buffer.erase(0, pos + 2);
                for (const auto& line : strings::split(frame, '\n'))
                {
                    auto trimmed = strings::trim(line);
                    if (strings::starts_with(trimmed, "event:"))
                    {
                        // Anthropic uses event: lines, just skip for now
                        continue;
                    }
                    if (strings::starts_with(trimmed, "data:"))
                    {
                        handle_event(strings::trim(trimmed.substr(5)));
                    }
                }
            }
            return !aborted;
        });

    if (!res)
    {
        throw LCError("Anthropic stream: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status / 100 != 2)
    {
        throw LCError("Anthropic stream: HTTP " + std::to_string(res->status));
    }

    ChatResponse out;
    out.model = cfg_.model;
    out.message = Message::assistant(aggregate_content);
    out.finish_reason = finish_reason.empty() ? std::string("stop") : finish_reason;
    if (out.finish_reason == "end_turn")
    {
        out.finish_reason = "stop";
    }
    return out;
}

} // namespace llm
} // namespace langchain
