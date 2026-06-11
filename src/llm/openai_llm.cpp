// src/llm/openai_llm.cpp -- OpenAI API backend.
#include "llm/openai_llm.h"
#include "llm/openai_common.h"

#include "util/logging.h"
#include "util/strings.h"

#include <httplib.h>
#include <spdlog/spdlog.h>

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

json content_part_to_openai_json(const ContentPart& part)
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

json message_to_openai_json(const Message& m)
{
    json j;
    j["role"] = to_string(m.role);

    if (!m.content_parts.empty())
    {
        json arr = json::array();
        for (const auto& part : m.content_parts)
        {
            arr.push_back(content_part_to_openai_json(part));
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

Message openai_json_to_message(const json& j)
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

json build_openai_payload(const ChatRequest& req,
                          const std::string& model,
                          bool stream,
                          const std::string& path)
{
    (void)path;
    json payload;
    payload["model"]  = model;
    payload["stream"] = stream;
    json msgs = json::array();
    for (const auto& m : req.messages)
    {
        msgs.push_back(message_to_openai_json(m));
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

OpenAILLM::OpenAILLM(OpenAILLMConfig cfg)
    : cfg_(std::move(cfg))
{
}

json OpenAILLM::build_payload(const ChatRequest& req, bool stream) const
{
    return build_openai_payload(req, cfg_.model, stream, "/v1/chat/completions");
}


ChatResponse OpenAILLM::invoke_impl(const ChatRequest& req)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers;
    if (!cfg_.api_key.empty())
    {
        headers.emplace("Authorization", "Bearer " + cfg_.api_key);
    }

    auto body = build_payload(req, false).dump();
    std::string path = make_chat_completions_path(parsed.path_prefix);
    LOG_INFO("OpenAI request: POST {} model={} body={}", path, cfg_.model, body);

    auto res = cli.Post(
        path.c_str(),
        headers,
        body,
        "application/json");

    if (!res)
    {
        LOG_ERROR("OpenAI request failed: {}", httplib::to_string(res.error()));
        throw LCError("OpenAI: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status / 100 != 2)
    {
        LOG_ERROR("OpenAI HTTP error: status={} body={}", res->status, res->body);
        throw LCError("OpenAI: HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    LOG_INFO("OpenAI response: status={} body={}", res->status, res->body);
    spdlog::default_logger()->flush();

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("OpenAI: invalid JSON response");
    }

    // Some providers wrap errors in HTTP 200 with a non-standard body shape
    // (e.g. {"code":500,"msg":"..."}). Detect and surface them.
    if (j.contains("code") && j.contains("msg"))
    {
        LOG_ERROR("OpenAI provider error: code={} msg={}",
                  j.value("code", 0), j.value("msg", std::string()));
        throw LCError("OpenAI: provider error (code=" +
                      std::to_string(j.value("code", 0)) + "): " +
                      j.value("msg", std::string()));
    }

    ChatResponse out;
    out.model = j.value("model", cfg_.model);
    const auto& choices = j.value("choices", json::array());
    if (!choices.empty())
    {
        const auto& c0 = choices[0];
        out.message = openai_json_to_message(c0.value("message", json::object()));
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

ChatResponse OpenAILLM::invoke_stream_impl(const ChatRequest& req,
                                           const StreamCallback& on_delta)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{
        {"Accept", "text/event-stream"}
    };
    if (!cfg_.api_key.empty())
    {
        headers.emplace("Authorization", "Bearer " + cfg_.api_key);
    }

    auto body = build_payload(req, true).dump();
    std::string path = make_chat_completions_path(parsed.path_prefix);
    LOG_INFO("OpenAI stream request: POST {} model={} body={}", path, cfg_.model, body);

    std::string aggregate_content;
    std::string finish_reason;
    std::vector<ToolCall> tool_calls;
    std::string buffer;
    bool aborted = false;

    auto handle_event = [&](const std::string& payload)
    {
        LOG_INFO("OpenAI stream raw payload: {}", payload);
        if (payload == "[DONE]")
        {
            return;
        }
        json j = json::parse(payload, nullptr, false);
        if (j.is_discarded())
        {
            LOG_INFO("OpenAI stream: failed to parse payload");
            return;
        }
        const auto& choices = j.value("choices", json::array());
        if (choices.empty())
        {
            LOG_INFO("OpenAI stream: choices empty");
            return;
        }
        const auto& c0 = choices[0];
        LOG_INFO("OpenAI stream choice: {}", c0.dump());
        if (c0.contains("delta"))
        {
            const auto& d = c0["delta"];
            LOG_INFO("OpenAI stream delta: {}", d.dump());
            if (d.contains("content") && d["content"].is_string())
            {
                std::string piece = d["content"].get<std::string>();
                aggregate_content += piece;
                LOG_INFO("OpenAI stream delta content: '{}', len={}", piece, piece.size());
                if (!on_delta(piece))
                {
                    aborted = true;
                }
            }
            else
            {
                LOG_INFO("OpenAI stream: no content in delta");
            }
            if (d.contains("tool_calls") && d["tool_calls"].is_array())
            {
                LOG_INFO("OpenAI stream delta tool_calls: {}", d["tool_calls"].dump());
                for (const auto& tc : d["tool_calls"])
                {
                    int index = tc.value("index", 0);
                    if (index >= static_cast<int>(tool_calls.size()))
                    {
                        tool_calls.resize(index + 1);
                    }
                    if (tc.contains("id") && tc["id"].is_string() && !tc["id"].get<std::string>().empty())
                    {
                        tool_calls[index].id = tc["id"].get<std::string>();
                    }
                    if (tc.contains("function"))
                    {
                        const auto& fn = tc["function"];
                        if (fn.contains("name") && fn["name"].is_string())
                        {
                            tool_calls[index].name = fn["name"].get<std::string>();
                        }
                        if (fn.contains("arguments") && fn["arguments"].is_string())
                        {
                            tool_calls[index].arguments += fn["arguments"].get<std::string>();
                        }
                    }
                }
            }
        }
        // Some providers (e.g. Kimi) return the full message in "message" field
        // instead of "delta" after tool_calls, especially on the final answer round.
        else if (c0.contains("message"))
        {
            const auto& msg = c0["message"];
            LOG_INFO("OpenAI stream message: {}", msg.dump());
            if (msg.contains("content") && msg["content"].is_string())
            {
                std::string piece = msg["content"].get<std::string>();
                aggregate_content += piece;
                LOG_INFO("OpenAI stream message content: '{}', len={}", piece, piece.size());
                if (!on_delta(piece))
                {
                    aborted = true;
                }
            }
            if (msg.contains("tool_calls") && msg["tool_calls"].is_array())
            {
                for (const auto& tc : msg["tool_calls"])
                {
                    int index = tc.value("index", 0);
                    if (index >= static_cast<int>(tool_calls.size()))
                    {
                        tool_calls.resize(index + 1);
                    }
                    if (tc.contains("id") && tc["id"].is_string() && !tc["id"].get<std::string>().empty())
                    {
                        tool_calls[index].id = tc["id"].get<std::string>();
                    }
                    if (tc.contains("function"))
                    {
                        const auto& fn = tc["function"];
                        if (fn.contains("name") && fn["name"].is_string())
                        {
                            tool_calls[index].name = fn["name"].get<std::string>();
                        }
                        if (fn.contains("arguments") && fn["arguments"].is_string())
                        {
                            tool_calls[index].arguments += fn["arguments"].get<std::string>();
                        }
                    }
                }
            }
        }
        if (c0.contains("finish_reason") && c0["finish_reason"].is_string())
        {
            finish_reason = c0["finish_reason"].get<std::string>();
            LOG_INFO("OpenAI stream finish_reason: {}", finish_reason);
        }
    };

    auto res = cli.Post(
        path.c_str(),
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
        LOG_ERROR("OpenAI stream request failed: {}", httplib::to_string(res.error()));
        throw LCError("OpenAI stream: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status / 100 != 2)
    {
        LOG_ERROR("OpenAI stream HTTP error: status={} body={}", res->status, res->body);
        throw LCError("OpenAI stream: HTTP " + std::to_string(res->status));
    }

    LOG_INFO("OpenAI stream response: status={} content_type={} body_len={} has_callback_data={}",
             res->status,
             res->get_header_value("Content-Type", "unknown"),
             res->body.size(),
             !buffer.empty() || aggregate_content.size() > 0);

    // Some providers wrap errors in HTTP 200 with a non-standard body shape.
    // For streaming, the error may arrive as a single SSE frame.
    // Also handle providers that return a full JSON response in body even when
    // stream=true (e.g. some providers after tool_calls round).
    if (!res->body.empty())
    {
        json j = json::parse(res->body, nullptr, false);
        if (!j.is_discarded())
        {
            if (j.contains("code") && j.contains("msg"))
            {
                LOG_ERROR("OpenAI stream provider error: code={} msg={}",
                          j.value("code", 0), j.value("msg", std::string()));
                throw LCError("OpenAI stream: provider error (code=" +
                              std::to_string(j.value("code", 0)) + "): " +
                              j.value("msg", std::string()));
            }
            // If the provider returned a full chat completion JSON in body
            // (non-SSE), extract content and tool_calls from it.
            if (j.contains("choices") && j["choices"].is_array())
            {
                const auto& choices = j["choices"];
                if (!choices.empty())
                {
                    const auto& c0 = choices[0];
                    if (c0.contains("message"))
                    {
                        const auto& msg = c0["message"];
                        LOG_INFO("OpenAI stream: provider returned full JSON in body, extracting content");
                        if (msg.contains("content") && msg["content"].is_string())
                        {
                            std::string piece = msg["content"].get<std::string>();
                            aggregate_content += piece;
                            LOG_INFO("OpenAI stream body content: '{}', len={}", piece, piece.size());
                            on_delta(piece);
                        }
                        if (msg.contains("tool_calls") && msg["tool_calls"].is_array())
                        {
                            for (const auto& tc : msg["tool_calls"])
                            {
                                int index = tc.value("index", 0);
                                if (index >= static_cast<int>(tool_calls.size()))
                                {
                                    tool_calls.resize(index + 1);
                                }
                                if (tc.contains("id") && tc["id"].is_string() && !tc["id"].get<std::string>().empty())
                                {
                                    tool_calls[index].id = tc["id"].get<std::string>();
                                }
                                if (tc.contains("function"))
                                {
                                    const auto& fn = tc["function"];
                                    if (fn.contains("name") && fn["name"].is_string())
                                    {
                                        tool_calls[index].name = fn["name"].get<std::string>();
                                    }
                                    if (fn.contains("arguments") && fn["arguments"].is_string())
                                    {
                                        tool_calls[index].arguments += fn["arguments"].get<std::string>();
                                    }
                                }
                            }
                        }
                        if (c0.contains("finish_reason") && c0["finish_reason"].is_string())
                        {
                            finish_reason = c0["finish_reason"].get<std::string>();
                        }
                    }
                }
            }
        }
    }

    LOG_INFO("OpenAI stream complete: content_len={} finish_reason={}",
              aggregate_content.size(), finish_reason);
    spdlog::default_logger()->flush();

    ChatResponse out;
    out.model = cfg_.model;
    out.message = Message::assistant(aggregate_content);
    out.message.tool_calls = std::move(tool_calls);
    out.finish_reason = finish_reason.empty() ? std::string("stop") : finish_reason;
    return out;
}

} // namespace llm
} // namespace langchain
