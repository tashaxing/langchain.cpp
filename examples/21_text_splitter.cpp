// examples/21_text_splitter.cpp — Text splitting strategies.
//
// Demonstrates:
//   1. CharacterTextSplitter: split on a fixed separator.
//   2. RecursiveCharacterTextSplitter: hierarchical splitting with overlap.
//   3. Custom length function (e.g., word count instead of character count).
//
// No network required.
#include "langchain.h"

#include <iostream>
#include <sstream>

std::string make_sample_text()
{
    return
        "C++20 introduces several major features that modernize the language.\n\n"
        "Modules replace header files and reduce compilation times significantly.\n\n"
        "Coroutines enable efficient asynchronous programming with co_await and co_yield.\n\n"
        "Concepts provide compile-time constraints on template parameters.\n\n"
        "Ranges unify algorithms and views under a single composable framework.\n\n"
        "std::format replaces printf and iostream formatting with Python-like syntax.\n\n"
        "std::span is a non-owning view over a contiguous sequence.\n\n"
        "std::jthread automatically joins on destruction, preventing thread leaks.";
}

int main()
{
    using namespace langchain;

    std::string text = make_sample_text();
    std::cout << "Original text: " << text.size() << " chars, ";

    // Count paragraphs
    int paragraphs = 0;
    std::size_t pos = 0;
    while ((pos = text.find("\n\n", pos)) != std::string::npos)
    {
        ++paragraphs;
        ++pos;
    }
    std::cout << paragraphs << " paragraphs\n\n";

    // ---- 1. CharacterTextSplitter ----
    std::cout << "=== CharacterTextSplitter (separator='\\n\\n', chunk_size=200) ===\n";
    text_splitter::CharacterTextSplitter::Config char_cfg;
    char_cfg.chunk_size = 200;
    char_cfg.chunk_overlap = 0;
    char_cfg.separator = "\n\n";
    text_splitter::CharacterTextSplitter char_splitter(char_cfg);

    auto char_chunks = char_splitter.split_text(text);
    std::cout << "Chunks: " << char_chunks.size() << "\n";
    for (std::size_t i = 0; i < char_chunks.size() && i < 3; ++i)
    {
        std::cout << "  Chunk " << i << " (" << char_chunks[i].size()
                  << " chars): " << char_chunks[i].substr(0, 60) << "...\n";
    }
    if (char_chunks.size() > 3)
    {
        std::cout << "  ... and " << (char_chunks.size() - 3) << " more\n";
    }

    // ---- 2. RecursiveCharacterTextSplitter ----
    std::cout << "\n=== RecursiveCharacterTextSplitter (chunk_size=150, overlap=30) ===\n";
    text_splitter::RecursiveCharacterTextSplitter::Config rec_cfg;
    rec_cfg.chunk_size = 150;
    rec_cfg.chunk_overlap = 30;
    text_splitter::RecursiveCharacterTextSplitter rec_splitter(rec_cfg);

    auto rec_chunks = rec_splitter.split_text(text);
    std::cout << "Chunks: " << rec_chunks.size() << "\n";
    for (std::size_t i = 0; i < rec_chunks.size() && i < 3; ++i)
    {
        std::cout << "  Chunk " << i << " (" << rec_chunks[i].size()
                  << " chars): " << rec_chunks[i].substr(0, 60) << "...\n";
    }
    if (rec_chunks.size() > 3)
    {
        std::cout << "  ... and " << (rec_chunks.size() - 3) << " more\n";
    }

    // ---- 3. Recursive with custom length function (word count) ----
    std::cout << "\n=== Recursive with word-count length (chunk_size=20 words) ===\n";
    text_splitter::RecursiveCharacterTextSplitter::Config word_cfg;
    word_cfg.chunk_size = 20;
    word_cfg.chunk_overlap = 5;
    word_cfg.length_func = [](const std::string& s) -> int
    {
        int count = 0;
        bool in_word = false;
        for (char c : s)
        {
            if (std::isspace(static_cast<unsigned char>(c)))
            {
                in_word = false;
            }
            else if (!in_word)
            {
                in_word = true;
                ++count;
            }
        }
        return count;
    };
    text_splitter::RecursiveCharacterTextSplitter word_splitter(word_cfg);

    auto word_chunks = word_splitter.split_text(text);
    std::cout << "Chunks: " << word_chunks.size() << "\n";
    for (std::size_t i = 0; i < word_chunks.size() && i < 3; ++i)
    {
        std::cout << "  Chunk " << i << " ("
                  << word_cfg.length_func(word_chunks[i]) << " words): "
                  << word_chunks[i].substr(0, 60) << "...\n";
    }
    if (word_chunks.size() > 3)
    {
        std::cout << "  ... and " << (word_chunks.size() - 3) << " more\n";
    }

    // ---- 4. Document splitting ----
    std::cout << "\n=== Document batch splitting ===\n";
    std::vector<Document> docs = {
        {"", "First document with some content.", {}, {}},
        {"", "Second document with different content.", {}, {}},
    };
    auto split_docs = rec_splitter.split_documents(docs);
    std::cout << "Split " << docs.size() << " documents into "
              << split_docs.size() << " chunks\n";

    return 0;
}
