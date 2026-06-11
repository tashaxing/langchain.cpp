// src/llm/gemini_llm.cpp -- Google Gemini API backend.
#include "llm/gemini_llm.h"

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

// Gemini uses a different message format - content as array of parts
json gemini_content_part_to_json(const ContentPart& part)
{
    if (part.type == "text")
    {
        return json{{"text", part.text}};
    }
    else if (part.type == "image_url")
    {
        // For Gemini, inline data with base64 is preferred
        // URL images need to be fetched separately, so we handle inline data
        if (part.url.find("data:") == 0)
        {
            auto comma = part.url.find(',');
            if (comma != std::string::npos)
            {
                std::string prefix = part.url.substr(0, comma);
                std::string data = part.url.substr(comma + 1);
                std::string mime = "image/png";
                auto semi = prefix.find(';');
                if (semi != std::string::npos)
                {
                    mime = prefix.substr(5, semi - 5);
                }
                return json{{"inlineData", {
                    {"mimeType", mime},
                    {"data", data}
                }}};
            }
        }
        // For non-data URLs, return as text reference
        return json{{"text", "[Image: " + part.url + "]"}};
    }
    else if (part.type == "image_base64")
    {
        return json{{"inlineData", {
            {"mimeType", part.mime_type},
            {"data", part.base64_data}
        }}};
    }
    return json{{"text", "[Unknown content type: " + part.type + "]"}};
}

json gemini_message_to_json(const Message& m)
{
    json j;
    j["role"] = (m.role == Role::User) ? "user" : "model";

    json parts = json::array();
    if (!m.content_parts.empty())
    {
        for (const auto& part : m.content_parts)
        {
            parts.push_back(gemini_content_part_to_json(part));
        }
    }
    else if (!m.content.empty())
    {
        parts.push_back({{"text", m.content}});
    }

    j["parts"] = std::move(parts);
    return j;
}

Message gemini_json_to_message(const json& j)
{
    Message m;
    std::string role = j.value("role", std::string("model"));
    m.role = (role == "user") ? Role::User : Role::Assistant;

    std::ostringstream oss;
    bool first = true;

    const auto& parts = j.value("parts", json::array());
    for (const auto& part : parts)
    {
        if (!first) oss << "\n";
        first = false;

        if (part.contains("text"))
        {
            std::string text = part.value("text", std::string());
            m.content_parts.push_back(ContentPart::text_part(text));
            oss << text;
        }
        else if (part.contains("inlineData"))
        {
            const auto& data = part["inlineData"];
            std::string mime = data.value("mimeType", "image/png");
            std::string base64 = data.value("data", std::string());
            m.content_parts.push_back(ContentPart::image_base64(base64, mime));
            oss << "[image_base64: " << mime << "]";
        }
    }

    m.content = oss.str();
    return m;
}

} // namespace

GeminiLLM::GeminiLLM(GeminiLLMConfig cfg)
    : cfg_(std::move(cfg))
{
}

json GeminiLLM::build_payload(const ChatRequest& req, bool stream) const
{
    json payload;

    // Gemini uses "contents" array with role+parts structure
    json contents = json::array();
    for (const auto& m : req.messages)
    {
        contents.push_back(gemini_message_to_json(m));
    }
    payload["contents"] = std::move(contents);

    // Generation config
    json generationConfig;
    if (req.temperature)
    {
        generationConfig["temperature"] = *req.temperature;
    }
    if (req.max_tokens)
    {
        generationConfig["maxOutputTokens"] = *req.max_tokens;
    }
    if (req.top_p)
    {
        generationConfig["topP"] = *req.top_p;
    }
    if (req.top_k)
    {
        generationConfig["topK"] = *req.top_k;
    }
    if (!req.stop.empty())
    {
        generationConfig["stopSequences"] = req.stop;
    }
    if (!generationConfig.empty())
    {
        payload["generationConfig"] = std::move(generationConfig);
    }

    // Safety settings (optional but recommended)
    json safetySettings = json::array({
        {{"category", "HARM_CATEGORY_HARASSMENT"}, {"threshold", "BLOCK_NONE"}},
        {{"category", "HARM_CATEGORY_HATE_SPEECH"}, {"threshold", "BLOCK_NONE"}},
        {{"category", "HARM_CATEGORY_SEXUALLY_EXPLICIT"}, {"threshold", "BLOCK_NONE"}},
        {{"category", "HARM_CATEGORY_DANGEROUS_CONTENT"}, {"threshold", "BLOCK_NONE"}}
    });
    payload["safetySettings"] = std::move(safetySettings);

    return payload;
}

Message GeminiLLM::decode_response(const json& j) const
{
    return gemini_json_to_message(j);
}

ChatResponse GeminiLLM::invoke_impl(const ChatRequest& req)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    // Build URL with API key as query parameter (Gemini style)
    std::string path = parsed.path_prefix + "/models/" + cfg_.model + ":generateContent";
    if (!cfg_.api_key.empty())
    {
        path += "?key=" + cfg_.api_key;
    }

    httplib::Headers headers{{"Content-Type", "application/json"}};

    auto body = build_payload(req, false).dump();
    auto res = cli.Post(path.c_str(), headers, body, "application/json");

    if (!res)
    {
        throw LCError("Gemini: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status / 100 != 2)
    {
        throw LCError("Gemini: HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("Gemini: invalid JSON response");
    }

    ChatResponse out;
    out.model = cfg_.model;

    // Check for candidates
    const auto& candidates = j.value("candidates", json::array());
    if (!candidates.empty())
    {
        const auto& c0 = candidates[0];
        out.message = decode_response(c0.value("content", json::object()));

        const auto& finish = c0.value("finishReason", std::string());
        if (finish == "STOP")
        {
            out.finish_reason = "stop";
        }
        else if (finish == "MAX_TOKENS")
        {
            out.finish_reason = "length";
        }
        else if (finish == "SAFETY")
        {
            out.finish_reason = "content_filter";
        }
        else
        {
            out.finish_reason = finish;
        }
    }

    if (j.contains("usageMetadata"))
    {
        const auto& u = j["usageMetadata"];
        out.usage.prompt_tokens     = u.value("promptTokenCount", 0);
        out.usage.completion_tokens = u.value("candidatesTokenCount", 0);
        out.usage.total_tokens      = u.value("totalTokenCount", 0);
    }

    return out;
}

ChatResponse GeminiLLM::invoke_stream_impl(const ChatRequest& req,
                                            const StreamCallback& on_delta)
{
    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    // Build URL with API key for streaming endpoint
    std::string path = parsed.path_prefix + "/models/" + cfg_.model + ":streamGenerateContent";
    if (!cfg_.api_key.empty())
    {
        path += "?key=" + cfg_.api_key;
    }

    httplib::Headers headers{
        {"Content-Type", "application/json"},
        {"Accept", "text/event-stream"}
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

        // Gemini streaming sends candidates similar to non-streaming
        const auto& candidates = j.value("candidates", json::array());
        if (!candidates.empty())
        {
            const auto& c0 = candidates[0];
            Message m = decode_response(c0.value("content", json::object()));
            if (!m.content.empty())
            {
                aggregate_content += m.content;
                if (!on_delta(m.content))
                {
                    aborted = true;
                }
            }

            const auto& finish = c0.value("finishReason", std::string());
            if (!finish.empty())
            {
                if (finish == "STOP")
                {
                    finish_reason = "stop";
                }
                else if (finish == "MAX_TOKENS")
                {
                    finish_reason = "length";
                }
                else if (finish == "SAFETY")
                {
                    finish_reason = "content_filter";
                }
                else
                {
                    finish_reason = finish;
                }
            }
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
            // Gemini uses newline-delimited JSON
            while ((pos = buffer.find('\n')) != std::string::npos)
            {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                line = strings::trim(line);
                if (!line.empty() && line != ",") // Sometimes Gemini sends commas between chunks
                {
                    // May have leading "data: " prefix or just JSON
                    if (strings::starts_with(line, "data:"))
                    {
                        line = strings::trim(line.substr(5));
                    }
                    handle_event(line);
                }
            }
            return !aborted;
        });

    if (!res)
    {
        throw LCError("Gemini stream: request failed: " + httplib::to_string(res.error()));
    }
    if (res->status / 100 != 2)
    {
        throw LCError("Gemini stream: HTTP " + std::to_string(res->status));
    }

    ChatResponse out;
    out.model = cfg_.model;
    out.message = Message::assistant(aggregate_content);
    out.finish_reason = finish_reason.empty() ? std::string("stop") : finish_reason;
    return out;
}

} // namespace llm
} // namespace langchain
