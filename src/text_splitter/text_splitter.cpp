// src/text_splitter/text_splitter.cpp -- text splitting implementations.
#include "text_splitter/text_splitter.h"


#include <algorithm>
#include <numeric>
#include <sstream>

namespace langchain
{
namespace text_splitter
{

// ---------------------------------------------------------------------------
// BaseSplitter -- default implementations
// ---------------------------------------------------------------------------

std::vector<Document> BaseSplitter::split_documents(const Document& doc) const
{
    auto chunks = split_text(doc.content);
    std::vector<Document> out;
    out.reserve(chunks.size());
    for (const auto& chunk : chunks)
    {
        Document d;
        d.id       = doc.id;
        d.content  = chunk;
        d.metadata = doc.metadata;
        out.push_back(std::move(d));
    }
    return out;
}

std::vector<Document> BaseSplitter::split_documents(const std::vector<Document>& docs) const
{
    std::vector<Document> out;
    for (const auto& doc : docs)
    {
        auto split = split_documents(doc);
        out.insert(out.end(), std::make_move_iterator(split.begin()),
                   std::make_move_iterator(split.end()));
    }
    return out;
}

// ---------------------------------------------------------------------------
// CharacterTextSplitter
// ---------------------------------------------------------------------------

CharacterTextSplitter::CharacterTextSplitter()
    : CharacterTextSplitter(Config{})
{
}

CharacterTextSplitter::CharacterTextSplitter(Config cfg)
    : cfg_(std::move(cfg))
{
    if (cfg_.chunk_size <= 0)
    {
        throw LCError("CharacterTextSplitter: chunk_size must be > 0");
    }
    if (cfg_.chunk_overlap < 0)
    {
        throw LCError("CharacterTextSplitter: chunk_overlap must be >= 0");
    }
    if (cfg_.chunk_overlap >= cfg_.chunk_size)
    {
        throw LCError("CharacterTextSplitter: chunk_overlap must be < chunk_size");
    }
}

std::vector<std::string> CharacterTextSplitter::split_text(const std::string& text) const
{
    if (text.empty())
    {
        return {};
    }

    auto parts = split_on_separator_(text);
    auto chunks = merge_chunks_(parts);

    return chunks;
}

std::vector<std::string> CharacterTextSplitter::split_on_separator_(
    const std::string& text) const
{
    if (cfg_.separator.empty())
    {
        return { text };
    }

    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start < text.size())
    {
        auto pos = text.find(cfg_.separator, start);
        if (pos == std::string::npos)
        {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, pos - start));
        start = pos + cfg_.separator.size();
    }
    return parts;
}

std::vector<std::string> CharacterTextSplitter::merge_chunks_(
    const std::vector<std::string>& chunks) const
{
    if (chunks.empty())
    {
        return {};
    }

    std::vector<std::string> out;
    std::string current;
    int current_len = 0;

    for (const auto& chunk : chunks)
    {
        int chunk_len = static_cast<int>(chunk.size());
        if (current.empty())
        {
            current = chunk;
            current_len = chunk_len;
        }
        else if (current_len + 1 + chunk_len <= cfg_.chunk_size)
        {
            current += cfg_.separator + chunk;
            current_len += static_cast<int>(cfg_.separator.size()) + chunk_len;
        }
        else
        {
            out.push_back(std::move(current));
            current = chunk;
            current_len = chunk_len;
        }
    }
    if (!current.empty())
    {
        out.push_back(std::move(current));
    }
    return out;
}

// ---------------------------------------------------------------------------
// RecursiveCharacterTextSplitter
// ---------------------------------------------------------------------------

RecursiveCharacterTextSplitter::RecursiveCharacterTextSplitter()
    : RecursiveCharacterTextSplitter(Config{})
{
}

RecursiveCharacterTextSplitter::RecursiveCharacterTextSplitter(Config cfg)
    : cfg_(std::move(cfg))
{
    if (cfg_.chunk_size <= 0)
    {
        throw LCError("RecursiveCharacterTextSplitter: chunk_size must be > 0");
    }
    if (cfg_.chunk_overlap < 0)
    {
        throw LCError("RecursiveCharacterTextSplitter: chunk_overlap must be >= 0");
    }
    if (cfg_.chunk_overlap >= cfg_.chunk_size)
    {
        throw LCError("RecursiveCharacterTextSplitter: chunk_overlap must be < chunk_size");
    }
}

int RecursiveCharacterTextSplitter::length_(const std::string& s) const
{
    if (cfg_.length_func)
    {
        return cfg_.length_func(s);
    }
    return static_cast<int>(s.size());
}

std::vector<std::string> RecursiveCharacterTextSplitter::split_text(const std::string& text) const
{
    if (text.empty())
    {
        return {};
    }
    auto chunks = split_single_(text, cfg_.separators, cfg_.chunk_size);
    if (cfg_.chunk_overlap > 0)
    {
        chunks = apply_overlap_(chunks);
    }
    return chunks;
}

std::vector<std::string> RecursiveCharacterTextSplitter::split_single_(
    const std::string& text,
    const std::vector<std::string>& separators,
    int chunk_size) const
{
    // If the text is short enough, return it as-is.
    if (length_(text) <= chunk_size)
    {
        return { text };
    }

    // No more separators to try; split into characters.
    if (separators.empty())
    {
        std::vector<std::string> out;
        out.reserve(static_cast<std::size_t>(text.size()) / chunk_size + 1);
        for (std::size_t i = 0; i < text.size(); i += static_cast<std::size_t>(chunk_size))
        {
            out.push_back(text.substr(i, static_cast<std::size_t>(chunk_size)));
        }
        return out;
    }

    // Try splitting with the first separator.
    const auto& separator = separators[0];
    auto remaining = separators;
    remaining.erase(remaining.begin());

    // If separator is empty, try next.
    if (separator.empty())
    {
        return split_single_(text, remaining, chunk_size);
    }

    // Split on the current separator.
    std::vector<std::string> parts;
    {
        std::size_t start = 0;
        while (start < text.size())
        {
            auto pos = text.find(separator, start);
            if (pos == std::string::npos)
            {
                parts.push_back(text.substr(start));
                break;
            }
            parts.push_back(text.substr(start, pos - start));
            start = pos + separator.size();
        }
    }

    // Now we have a list of parts. Merge them into chunks that fit chunk_size,
    // then recurse on chunks that are still too long with the remaining separators.
    std::vector<std::string> merged = merge_chunks_(parts, chunk_size);

    std::vector<std::string> out;
    out.reserve(merged.size());
    for (const auto& chunk : merged)
    {
        if (length_(chunk) <= chunk_size)
        {
            out.push_back(chunk);
        }
        else
        {
            auto sub = split_single_(chunk, remaining, chunk_size);
            out.insert(out.end(), sub.begin(), sub.end());
        }
    }
    return out;
}

std::vector<std::string> RecursiveCharacterTextSplitter::merge_chunks_(
    const std::vector<std::string>& chunks,
    int chunk_size) const
{
    if (chunks.empty())
    {
        return {};
    }

    // Try to merge adjacent chunks until we hit chunk_size, then start a new one.
    std::vector<std::string> out;
    std::string current;
    int current_len = 0;

    for (const auto& chunk : chunks)
    {
        int chunk_len = length_(chunk);
        if (current.empty())
        {
            current = chunk;
            current_len = chunk_len;
        }
        else if (current_len + 1 + chunk_len <= chunk_size)
        {
            // Merge with separator (space) in between.
            current += " " + chunk;
            current_len += 1 + chunk_len;
        }
        else
        {
            out.push_back(std::move(current));
            current = chunk;
            current_len = chunk_len;
        }
    }
    if (!current.empty())
    {
        out.push_back(std::move(current));
    }
    return out;
}

std::vector<std::string> RecursiveCharacterTextSplitter::apply_overlap_(
    const std::vector<std::string>& chunks) const
{
    if (chunks.size() < 2 || cfg_.chunk_overlap <= 0)
    {
        return chunks;
    }

    std::vector<std::string> out;
    out.push_back(chunks[0]);

    for (std::size_t i = 1; i < chunks.size(); ++i)
    {
        const auto& prev = chunks[i - 1];
        const auto& curr = chunks[i];

        // Append overlap from the end of the previous chunk.
        std::string merged;
        if (static_cast<int>(prev.size()) > cfg_.chunk_overlap)
        {
            merged = prev.substr(prev.size() - cfg_.chunk_overlap);
            merged += curr;
        }
        else
        {
            merged = curr;
        }
        out.push_back(std::move(merged));
    }

    return out;
}

} // namespace text_splitter
} // namespace langchain
