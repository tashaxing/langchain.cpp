#include "embedding/embedding.h"

#include <httplib.h>

#include <cmath>
#include <cstdint>

namespace langchain
{
namespace embedding
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

} // namespace

// ---------------------------------------------------------------------------
// HttpEmbedding
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> HttpEmbedding::embed_documents(
    const std::vector<std::string>& texts)
{
    if (texts.empty())
    {
        return {};
    }

    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{{"Content-Type", "application/json"}};
    if (!cfg_.api_key.empty())
    {
        headers.emplace("Authorization", "Bearer " + cfg_.api_key);
    }

    json payload;
    payload["model"] = cfg_.model;
    payload["input"] = texts;

    auto target = parsed.path_prefix + cfg_.path;
    auto res = cli.Post(target.c_str(), headers, payload.dump(), "application/json");
    if (!res)
    {
        throw LCError("HttpEmbedding: request failed");
    }
    if (res->status / 100 != 2)
    {
        throw LCError("HttpEmbedding: HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("HttpEmbedding: bad JSON");
    }
    if (!j.is_object())
    {
        throw LCError("HttpEmbedding: unexpected JSON type (expected object): " + j.dump());
    }

    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    const auto& data = j.value("data", json::array());
    for (const auto& item : data)
    {
        if (!item.is_object())
        {
            throw LCError("HttpEmbedding: unexpected item type in data array (expected object): " + item.dump());
        }
        std::vector<float> v;
        const auto& emb = item.value("embedding", json::array());
        for (const auto& x : emb)
        {
            v.push_back(x.get<float>());
        }
        out.push_back(std::move(v));
    }
    return out;
}

// ---------------------------------------------------------------------------
// OllamaEmbedding
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> OllamaEmbedding::embed_documents(
    const std::vector<std::string>& texts)
{
    if (texts.empty())
    {
        return {};
    }

    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    std::vector<std::vector<float>> out;
    out.reserve(texts.size());

    // Ollama /api/embeddings handles one text at a time.
    for (const auto& text : texts)
    {
        json payload;
        payload["model"] = cfg_.model;
        payload["prompt"] = text;

        auto res = cli.Post(
            "/api/embeddings",
            httplib::Headers{{"Content-Type", "application/json"}},
            payload.dump(),
            "application/json");

        if (!res)
        {
            throw LCError("OllamaEmbedding: request failed");
        }
        if (res->status / 100 != 2)
        {
            throw LCError("OllamaEmbedding: HTTP " + std::to_string(res->status));
        }

        json j = json::parse(res->body, nullptr, false);
        if (j.is_discarded())
        {
            throw LCError("OllamaEmbedding: bad JSON");
        }
        if (!j.is_object())
        {
            throw LCError("OllamaEmbedding: unexpected JSON type (expected object): " + j.dump());
        }

        std::vector<float> vec;
        for (const auto& x : j.value("embedding", json::array()))
        {
            vec.push_back(x.get<float>());
        }
        out.push_back(std::move(vec));
    }

    return out;
}

// ---------------------------------------------------------------------------
// LocalAIEmbedding
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> LocalAIEmbedding::embed_documents(
    const std::vector<std::string>& texts)
{
    if (texts.empty())
    {
        return {};
    }

    auto parsed = split_base_url(cfg_.base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_connection_timeout(cfg_.connect_timeout_sec);
    cli.set_read_timeout(cfg_.read_timeout_sec);

    httplib::Headers headers{{"Content-Type", "application/json"}};
    if (!cfg_.api_key.empty())
    {
        headers.emplace("Authorization", "Bearer " + cfg_.api_key);
    }

    json payload;
    payload["model"] = cfg_.model;
    payload["input"] = texts;

    auto target = parsed.path_prefix + cfg_.path;
    auto res = cli.Post(target.c_str(), headers, payload.dump(), "application/json");
    if (!res)
    {
        throw LCError("LocalAIEmbedding: request failed");
    }
    if (res->status / 100 != 2)
    {
        throw LCError("LocalAIEmbedding: HTTP " + std::to_string(res->status) + ": " + res->body);
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("LocalAIEmbedding: bad JSON");
    }
    if (!j.is_object())
    {
        throw LCError("LocalAIEmbedding: unexpected JSON type (expected object): " + j.dump());
    }

    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& item : j.value("data", json::array()))
    {
        std::vector<float> vec;
        for (const auto& x : item.value("embedding", json::array()))
        {
            vec.push_back(x.get<float>());
        }
        out.push_back(std::move(vec));
    }
    return out;
}

// ---------------------------------------------------------------------------
// HashingEmbedding
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> HashingEmbedding::embed_documents(
    const std::vector<std::string>& texts)
{
    // FNV-1a over rolling 3-grams; sum into a fixed-dim vector then L2-normalize.
    auto fnv1a = [](const char* data, std::size_t n)
    {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= static_cast<unsigned char>(data[i]);
            h *= 1099511628211ULL;
        }
        return h;
    };

    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& t : texts)
    {
        std::vector<float> v(static_cast<std::size_t>(dim_), 0.0f);
        if (t.size() >= 3)
        {
            for (std::size_t i = 0; i + 3 <= t.size(); ++i)
            {
                std::uint64_t h = fnv1a(t.data() + i, 3);
                std::size_t bucket =
                    static_cast<std::size_t>(h % static_cast<std::uint64_t>(dim_));
                v[bucket] += 1.0f;
            }
        }
        double norm = 0.0;
        for (float x : v)
        {
            norm += static_cast<double>(x) * x;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0)
        {
            for (float& x : v)
            {
                x = static_cast<float>(x / norm);
            }
        }
        out.push_back(std::move(v));
    }
    return out;
}

} // namespace embedding
} // namespace langchain
