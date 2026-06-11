// langchain/text_splitter/text_splitter.h
// Text splitting abstractions: BaseSplitter + CharacterTextSplitter + RecursiveCharacterTextSplitter.
#pragma once

#include "util/common.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace text_splitter
{

// ---------------------------------------------------------------------------
// BaseSplitter -- abstract interface for all text splitters.
// ---------------------------------------------------------------------------
class BaseSplitter
{
public:
    virtual ~BaseSplitter() = default;

    // Split a single text into chunks.
    virtual std::vector<std::string> split_text(const std::string& text) const = 0;

    // Split a single Document; each chunk becomes a new Document with the
    // same metadata.
    virtual std::vector<Document> split_documents(const Document& doc) const;

    // Batch variant for multiple documents.
    virtual std::vector<Document> split_documents(const std::vector<Document>& docs) const;
};

using SplitterPtr = std::shared_ptr<BaseSplitter>;

// ---------------------------------------------------------------------------
// CharacterTextSplitter
// ---------------------------------------------------------------------------
// Simple single-separator splitter: splits on a given separator and optionally
// joins adjacent chunks until they hit chunk_size.
// ---------------------------------------------------------------------------
class CharacterTextSplitter : public BaseSplitter
{
public:
    struct Config
    {
        // Target chunk size in characters.
        int chunk_size = 1000;

        // Overlap between consecutive chunks.
        int chunk_overlap = 200;

        // Separator to split on (e.g. "\n\n").
        std::string separator = "\n\n";

        Config() = default;
        explicit Config(int cs) : chunk_size(cs) {}
    };

    CharacterTextSplitter();
    explicit CharacterTextSplitter(Config cfg);

    std::vector<std::string> split_text(const std::string& text) const override;

private:
    Config cfg_;

    std::vector<std::string> split_on_separator_(const std::string& text) const;
    std::vector<std::string> merge_chunks_(
        const std::vector<std::string>& chunks) const;
};

// ---------------------------------------------------------------------------
// RecursiveCharacterTextSplitter
// ---------------------------------------------------------------------------
// LangChain-style recursive text splitter: tries each separator in order,
// splits on it, then recurses into chunks that are still too long.
// Default separators (largest to smallest):
//   "\n\n" -> "\n" -> "." -> "!" -> "?" -> " " -> ""
// ---------------------------------------------------------------------------
class RecursiveCharacterTextSplitter : public BaseSplitter
{
public:
    struct Config
    {
        // Target chunk size in characters.
        int chunk_size = 1000;

        // Overlap between consecutive chunks (helps preserve context).
        int chunk_overlap = 200;

        // Separators to try, in order (largest to smallest structural unit).
        std::vector<std::string> separators = { "\n\n", "\n", ".", "!", "?", " ", "" };

        // Optional custom length function (e.g., token count).
        // If null, defaults to character count.
        std::function<int(const std::string&)> length_func;

        Config()
        {
        }
        explicit Config(int cs) : chunk_size(cs)
        {
        }

        // Rule of five: explicitly default copy/move.
        Config(const Config&) = default;
        Config& operator=(const Config&) = default;
        Config(Config&&) = default;
        Config& operator=(Config&&) = default;
    };

    RecursiveCharacterTextSplitter();
    explicit RecursiveCharacterTextSplitter(Config cfg);

    std::vector<std::string> split_text(const std::string& text) const override;

private:
    Config cfg_;

    // Core recursive splitting logic.
    std::vector<std::string> split_single_(const std::string& text,
                                           const std::vector<std::string>& separators,
                                           int chunk_size) const;

    // Apply chunk overlap: given a list of chunks, add overlap between them.
    std::vector<std::string> apply_overlap_(const std::vector<std::string>& chunks) const;

    // Length of a string according to the configured length function.
    int length_(const std::string& s) const;

    // Merge short chunks with the next one until it fits the size.
    std::vector<std::string> merge_chunks_(const std::vector<std::string>& chunks,
                                           int chunk_size) const;
};

} // namespace text_splitter
} // namespace langchain
