// src/retriever/retriever.cpp -- retriever implementations.
#include "retriever/retriever.h"


#include <algorithm>
#include <cmath>
#include <cctype>
#include <locale>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace langchain
{
namespace retriever
{

// ---------------------------------------------------------------------------
// VectorStoreRetriever
// ---------------------------------------------------------------------------
VectorStoreRetriever::VectorStoreRetriever(vectorstore::VectorStorePtr vs, int default_k)
    : vs_(std::move(vs)),
      default_k_(default_k)
{
}

std::vector<vectorstore::ScoredDocument> VectorStoreRetriever::retrieve(
    const std::string& query, int k)
{
    if (!vs_)
    {
        return {};
    }
    int topk = (k > 0) ? k : default_k_;
    return vs_->similarity_search(query, topk);
}

// ---------------------------------------------------------------------------
// BM25Retriever
// ---------------------------------------------------------------------------

namespace
{

// Simple tokenization: lowercase, split on non-alnum.
std::vector<std::string> tokenize(const std::string& text)
{
    std::vector<std::string> tokens;
    std::string current;
    for (unsigned char c : text)
    {
        if (std::isalnum(c))
        {
            current += static_cast<char>(std::tolower(c));
        }
        else if (!current.empty())
        {
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty())
    {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

} // namespace

struct BM25Retriever::Impl
{
    // doc_freqs[term][doc_id] = count
    std::unordered_map<std::string, std::unordered_map<std::size_t, float>> doc_freqs;
    std::unordered_map<std::string, std::size_t> doc_counts;
    std::vector<Document> docs;
    float avgdl = 0.0f;
    float k1 = 1.5f;
    float b = 0.75f;
    std::size_t total_tokens = 0;
};

BM25Retriever::~BM25Retriever() = default;

BM25Retriever::BM25Retriever(const std::vector<Document>& docs,
                             float k1,
                             float b)
    : impl_(new Impl())
{
    impl_->k1 = k1;
    impl_->b = b;
    impl_->docs = docs;

    std::size_t total_tokens = 0;
    for (std::size_t i = 0; i < docs.size(); ++i)
    {
        auto tokens = tokenize(docs[i].content);
        total_tokens += tokens.size();
        std::unordered_set<std::string> seen;
        for (const auto& t : tokens)
        {
            impl_->doc_freqs[t][i] += 1.0f;
            seen.insert(t);
        }
        for (const auto& t : seen)
        {
            impl_->doc_counts[t]++;
        }
    }

    impl_->avgdl = docs.empty() ? 0.0f : static_cast<float>(total_tokens) / static_cast<float>(docs.size());
    impl_->total_tokens = total_tokens;
}

std::vector<vectorstore::ScoredDocument> BM25Retriever::retrieve(
    const std::string& query, int k)
{
    if (impl_->docs.empty() || k <= 0)
    {
        return {};
    }

    auto qtokens = tokenize(query);
    if (qtokens.empty())
    {
        return {};
    }

    std::unordered_map<std::size_t, float> scores;
    float N = static_cast<float>(impl_->docs.size());

    for (const auto& t : qtokens)
    {
        auto it = impl_->doc_freqs.find(t);
        if (it == impl_->doc_freqs.end())
        {
            continue;
        }

        float df = static_cast<float>(impl_->doc_counts[t]);
        float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));

        for (const auto& kv : it->second)
        {
            std::size_t doc_idx = kv.first;
            float tf = kv.second;
            float dl = static_cast<float>(tokenize(impl_->docs[doc_idx].content).size());
            float denom = tf + impl_->k1 * (1.0f - impl_->b + impl_->b * dl / impl_->avgdl);
            scores[doc_idx] += idf * (tf * (impl_->k1 + 1.0f)) / denom;
        }
    }

    std::vector<vectorstore::ScoredDocument> out;
    out.reserve(scores.size());
    for (const auto& kv : scores)
    {
        out.push_back(vectorstore::ScoredDocument{impl_->docs[kv.first], kv.second});
    }

    int top = std::min<int>(k, static_cast<int>(out.size()));
    std::partial_sort(out.begin(), out.begin() + top, out.end(),
                      [](const vectorstore::ScoredDocument& a,
                         const vectorstore::ScoredDocument& b)
                      {
                          return a.score > b.score;
                      });
    out.resize(static_cast<std::size_t>(top));
    return out;
}

// ---------------------------------------------------------------------------
// MultiQueryRetriever
// ---------------------------------------------------------------------------
MultiQueryRetriever::MultiQueryRetriever(RetrieverPtr base,
                                         std::vector<std::string> (*expander)(const std::string&))
    : base_(std::move(base)),
      expander_(expander)
{
}

std::vector<vectorstore::ScoredDocument> MultiQueryRetriever::retrieve(
    const std::string& query, int k)
{
    if (!base_)
    {
        return {};
    }

    std::vector<std::string> queries;
    if (expander_)
    {
        queries = expander_(query);
    }
    else
    {
        queries = { query };
    }

    // Deduplicate by doc id, keep max score.
    std::unordered_map<std::string, vectorstore::ScoredDocument> merged;
    for (const auto& q : queries)
    {
        auto hits = base_->retrieve(q, k);
        for (const auto& h : hits)
        {
            auto it = merged.find(h.doc.id);
            if (it == merged.end() || h.score > it->second.score)
            {
                merged[h.doc.id] = h;
            }
        }
    }

    std::vector<vectorstore::ScoredDocument> out;
    out.reserve(merged.size());
    for (auto& kv : merged)
    {
        out.push_back(std::move(kv.second));
    }
    std::sort(out.begin(), out.end(),
              [](const vectorstore::ScoredDocument& a,
                 const vectorstore::ScoredDocument& b)
              {
                  return a.score > b.score;
              });

    if (static_cast<int>(out.size()) > k)
    {
        out.resize(static_cast<std::size_t>(k));
    }
    return out;
}

} // namespace retriever
} // namespace langchain
