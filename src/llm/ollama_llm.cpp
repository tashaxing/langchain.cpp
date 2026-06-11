// src/llm/ollama_llm.cpp -- Ollama local API backend.
#include "llm/ollama_llm.h"

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

// Ollama uses OpenAI-compatible format but with different structure
json ollama_message_to_json(const Message& m)
{
    json j;
    j["role"] = to_string(m.role);

    if (!m.content_parts.empty())
    {
        // For multimodal, Ollama uses images array
        std::vector<std::string> images;
        std::string text_content;
        for (const auto& part : m.content_parts)
        {
            if (part.type == "text")
            {
                text_content += part.text;
            }
            else if (part.type == "image_url")
            {
                images.push_back(part.url);
            }
            else if (part.type == "image_base64")
            {
                images.push_back(part.base64_data);
            }
        }
        j["content"] = text_content;
        if (!images.empty())
        {
            j["images"] = std::move(images);
        }
    }
    else if (!m.content.empty())
    {
        j["content"] = m.content;
    }

    return j;
}

Message ollama_json_to_message(const json& j)
{
    Message m;
    m.role = role_from_string(j.value("role", std::string("assistant")));

    if (j.contains("content"))
    {
        m.content = j.value("content", std::string());
    }

    // Ollama may return "message" nested structure
    if (j.contains("message") && j["message"].is_object())
    {
        const auto& msg = j["message"];
        m.role = role_from_string(msg.value("role", std::string("assistant")));
        m.content = msg.value("content", std::string());
    }

    return m;
}

} // namespace

OllamaLLM::OllamaLLM(OllamaLLMConfig cfg)
    : cfg_(std::move(cfg))
{
}

json OllamaLLM::build_payload(const ChatRequest& req, bool stream) const
{
    json payload;
    payload["model"] = cfg_.model;
    payload["stream"] = stream;

    json msgs = json::array();
    for (const auto& m : req.messages)
    {
        msgs.push_back(ollama_message_to_json(m));
    }
    payload["messages"] = std::move(msgs);

    // Ollama options
    json options = json::object();
    if (req.temperature)
    {
        options["temperature"] = *req.temperature;
    }
    if (req.max_tokens)
    {
        options["num_predict"] = *req.max_tokens;
    }
    if (req.top_p)
    {
        options["top_p"] = *req.top_p;
    }
    if (req.top_k)
    {
        options["top_k"] = *req.top_k;
    }
    if (!req.stop.empty())
    {
        options["stop"] = req.stop;
    }
    if (!options.empty())
    {
        payload["options"] = std::move(options);
    }

    return payload;
}

Message OllamaLLM::decode_response(const json& j) const
{
    return ollama_json_to_message(j);
}

std::vector<std::string> OllamaLLM::list_models() const
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    auto res = cli.Get((parsed.path_prefix + "/api/tags").c_str());
    if (!res)
    {
        throw LCError("Ollama: list models failed: " + httplib::to_string(res.error()));
    }
    if (res->status != 200)
    {
        throw LCError("Ollama: list models HTTP " + std::to_string(res->status));
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("Ollama: invalid JSON response");
    }

    std::vector<std::string> models;
    const auto& arr = j.value("models", json::array());
    for (const auto& m : arr)
    {
        models.push_back(m.value("name", std::string()));
    }
    return models;
}

void OllamaLLM::pull_model(const std::string& model) const
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);
    // Pull can take a long time, use longer timeout
    cli.set_read_timeout(600);

    json payload;
    payload["name"] = model;

    auto res = cli.Post(
        (parsed.path_prefix + "/api/pull").c_str(),
        payload.dump(),
        "application/json");

    if (!res)
    {
        throw LCError("Ollama: pull model failed: " + httplib::to_string(res.error()));
    }
    if (res->status != 200)
    {
        throw LCError("Ollama: pull model HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    // Check response for success
    json j = json::parse(res->body, nullptr, false);
    if (!j.is_discarded() && j.contains("status"))
    {
        std::string status = j.value("status", std::string());
        if (status.find("success") == std::string::npos &&
            status.find("pulling") == std::string::npos)
        {
            throw LCError("Ollama: pull model failed: " + status);
        }
    }
}

ChatResponse OllamaLLM::invoke_impl(const ChatRequest& req)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{{"Content-Type", "application/json"}};
    if (!cfg_.api_key.empty())
    {
        headers.emplace("Authorization", "Bearer " + cfg_.api_key);
    }

    auto body = build_payload(req, false).dump();
    auto res = cli.Post(
        (parsed.path_prefix + "/api/chat").c_str(),
        headers,
        body,
        "application/json");

    if (!res)
    {
        throw LCError("Ollama: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status != 200)
    {
        throw LCError("Ollama: HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("Ollama: invalid JSON response");
    }

    ChatResponse out;
    out.model = j.value("model", cfg_.model);
    out.message = decode_response(j);
    out.finish_reason = j.value("done", false) ? "stop" : "";

    if (j.contains("prompt_eval_count"))
    {
        out.usage.prompt_tokens = j.value("prompt_eval_count", 0);
    }
    if (j.contains("eval_count"))
    {
        out.usage.completion_tokens = j.value("eval_count", 0);
    }
    out.usage.total_tokens = out.usage.prompt_tokens + out.usage.completion_tokens;

    return out;
}

ChatResponse OllamaLLM::invoke_stream_impl(const ChatRequest& req,
                                           const StreamCallback& on_delta)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{{"Content-Type", "application/json"}};
    if (!cfg_.api_key.empty())
    {
        headers.emplace("Authorization", "Bearer " + cfg_.api_key);
    }

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

        if (j.contains("message") && j["message"].is_object())
        {
            const auto& msg = j["message"];
            if (msg.contains("content"))
            {
                std::string piece = msg.value("content", std::string());
                aggregate_content += piece;
                if (!piece.empty() && !on_delta(piece))
                {
                    aborted = true;
                }
            }
        }
        else if (j.contains("response"))
        {
            // Legacy format
            std::string piece = j.value("response", std::string());
            aggregate_content += piece;
            if (!piece.empty() && !on_delta(piece))
            {
                aborted = true;
            }
        }

        if (j.value("done", false))
        {
            finish_reason = "stop";
        }
    };

    auto res = cli.Post(
        (parsed.path_prefix + "/api/chat").c_str(),
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
            // Ollama uses newline-delimited JSON (NDJSON)
            while ((pos = buffer.find('\n')) != std::string::npos)
            {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                if (!line.empty())
                {
                    handle_event(line);
                }
            }
            return !aborted;
        });

    if (!res)
    {
        throw LCError("Ollama stream: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status != 200)
    {
        throw LCError("Ollama stream: HTTP " + std::to_string(res->status));
    }

    ChatResponse out;
    out.model = cfg_.model;
    out.message = Message::assistant(aggregate_content);
    out.finish_reason = finish_reason.empty() ? std::string("stop") : finish_reason;
    return out;
}

} // namespace llm
} // namespace langchain
