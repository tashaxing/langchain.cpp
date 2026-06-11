#include "vectorstore/vectorstore.h"
#include "util/logging.h"

#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <sstream>

#ifdef LC_HAS_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#endif

namespace langchain
{
namespace vectorstore
{

namespace
{

std::string gen_id()
{
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::ostringstream oss;
    oss << std::hex << dist(rng);
    return oss.str();
}

float cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty())
    {
        return 0.0f;
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0.0 || nb == 0.0)
    {
        return 0.0f;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

class SqliteStmt
{
public:
    explicit SqliteStmt(sqlite3_stmt* stmt)
        : stmt_(stmt)
    {
    }

    ~SqliteStmt()
    {
        if (stmt_)
        {
            sqlite3_finalize(stmt_);
        }
    }

    SqliteStmt(const SqliteStmt&) = delete;
    SqliteStmt& operator=(const SqliteStmt&) = delete;

    sqlite3_stmt* get() const
    {
        return stmt_;
    }

private:
    sqlite3_stmt* stmt_;
};

class SqliteTransaction
{
public:
    explicit SqliteTransaction(sqlite3* db)
        : db_(db)
    {
        if (sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr) == SQLITE_OK)
        {
            active_ = true;
        }
    }

    ~SqliteTransaction()
    {
        if (active_)
        {
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        }
    }

    SqliteTransaction(const SqliteTransaction&) = delete;
    SqliteTransaction& operator=(const SqliteTransaction&) = delete;

    void commit()
    {
        if (active_)
        {
            sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
            active_ = false;
        }
    }

private:
    sqlite3* db_ = nullptr;
    bool     active_ = false;
};

} // namespace

// ---------------- InMemoryVectorStore ----------------

std::vector<std::string> InMemoryVectorStore::add_documents(std::vector<Document> docs)
{
    std::vector<std::string> texts;
    texts.reserve(docs.size());
    for (auto& d : docs)
    {
        if (d.id.empty())
        {
            d.id = gen_id();
        }
        texts.push_back(d.content);
    }
    auto vecs = embedder_->embed_documents(texts);
    std::vector<std::string> ids;
    ids.reserve(docs.size());

    std::lock_guard<std::mutex> lk(mu_);
    for (std::size_t i = 0; i < docs.size(); ++i)
    {
        docs[i].embedding = std::move(vecs[i]);
        ids.push_back(docs[i].id);
        docs_.push_back(std::move(docs[i]));
    }
    return ids;
}

std::vector<ScoredDocument> InMemoryVectorStore::similarity_search(
    const std::string& query, int k,
    const std::unordered_map<std::string, std::string>& filter)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (docs_.empty() || k <= 0)
    {
        return {};
    }
    auto q = embedder_->embed_query(query);
    std::vector<ScoredDocument> scored;
    scored.reserve(docs_.size());
    for (const auto& d : docs_)
    {
        if (!filter.empty())
        {
            bool match = true;
            for (const auto& kv : filter)
            {
                auto it = d.metadata.find(kv.first);
                if (it == d.metadata.end() || it->second != kv.second)
                {
                    match = false;
                    break;
                }
            }
            if (!match)
            {
                continue;
            }
        }
        scored.push_back(ScoredDocument{d, cosine(q, d.embedding)});
    }
    int top = std::min<int>(k, static_cast<int>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + top, scored.end(),
                      [](const ScoredDocument& a, const ScoredDocument& b)
                      {
                          return a.score > b.score;
                      });
    scored.resize(static_cast<std::size_t>(top));
    return scored;
}

std::size_t InMemoryVectorStore::delete_documents(const std::vector<std::string>& ids)
{
    std::lock_guard<std::mutex> lk(mu_);
    std::size_t before = docs_.size();
    for (const auto& id : ids)
    {
        auto it = std::remove_if(docs_.begin(), docs_.end(),
                                 [&id](const Document& d) { return d.id == id; });
        docs_.erase(it, docs_.end());
    }
    return before - docs_.size();
}

void InMemoryVectorStore::update_document(const std::string& id, Document doc)
{
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& d : docs_)
    {
        if (d.id == id)
        {
            auto vec = embedder_->embed_documents({doc.content});
            d.content = std::move(doc.content);
            d.metadata = std::move(doc.metadata);
            d.embedding = std::move(vec[0]);
            return;
        }
    }
    throw LCError("InMemoryVectorStore: document not found: " + id);
}

std::size_t InMemoryVectorStore::size() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return docs_.size();
}

// ---------------- SqliteVectorStore ----------------

struct SqliteVectorStore::Impl
{
    sqlite3* db = nullptr;
};

SqliteVectorStore::SqliteVectorStore(embedding::EmbeddingPtr embedder,
                                     const std::string& db_path)
    : InMemoryVectorStore(std::move(embedder)),
      impl_(std::make_unique<Impl>())
{
    open_(db_path);
    load_all_();
}

SqliteVectorStore::~SqliteVectorStore()
{
    if (impl_ && impl_->db)
    {
        sqlite3_close(impl_->db);
    }
}

void SqliteVectorStore::open_(const std::string& path)
{
    if (sqlite3_open(path.c_str(), &impl_->db) != SQLITE_OK)
    {
        std::string err = sqlite3_errmsg(impl_->db);
        throw LCError("SqliteVectorStore: open failed: " + err);
    }
    const char* schema =
        "CREATE TABLE IF NOT EXISTS documents ("
        " id TEXT PRIMARY KEY,"
        " content TEXT NOT NULL,"
        " metadata TEXT,"
        " embedding BLOB NOT NULL);";
    char* errmsg = nullptr;
    if (sqlite3_exec(impl_->db, schema, nullptr, nullptr, &errmsg) != SQLITE_OK)
    {
        std::string e = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw LCError("SqliteVectorStore: schema init failed: " + e);
    }
}

void SqliteVectorStore::load_all_()
{
    sqlite3_stmt* raw_stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db,
                           "SELECT id, content, metadata, embedding FROM documents;",
                           -1, &raw_stmt, nullptr) != SQLITE_OK)
    {
        throw LCError("SqliteVectorStore: prepare load failed");
    }
    SqliteStmt stmt(raw_stmt);
    while (sqlite3_step(stmt.get()) == SQLITE_ROW)
    {
        Document d;
        d.id      = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 0));
        d.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 1));
        const unsigned char* meta_txt = sqlite3_column_text(stmt.get(), 2);
        if (meta_txt)
        {
            auto j = json::parse(reinterpret_cast<const char*>(meta_txt), nullptr, false);
            if (!j.is_discarded() && j.is_object())
            {
                for (auto it = j.begin(); it != j.end(); ++it)
                {
                    d.metadata[it.key()] = it.value().is_string()
                                               ? it.value().get<std::string>()
                                               : it.value().dump();
                }
            }
        }
        const void* blob = sqlite3_column_blob(stmt.get(), 3);
        int blob_bytes   = sqlite3_column_bytes(stmt.get(), 3);
        std::size_t n = static_cast<std::size_t>(blob_bytes) / sizeof(float);
        d.embedding.resize(n);
        if (blob && n > 0)
        {
            std::memcpy(d.embedding.data(), blob, n * sizeof(float));
        }
        docs_.push_back(std::move(d));
    }
}

std::vector<std::string> SqliteVectorStore::add_documents(std::vector<Document> docs)
{
    auto ids = InMemoryVectorStore::add_documents(std::move(docs));

    sqlite3_stmt* raw_stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db,
                           "INSERT OR REPLACE INTO documents(id, content, metadata, embedding)"
                           " VALUES(?,?,?,?);",
                           -1, &raw_stmt, nullptr) != SQLITE_OK)
    {
        throw LCError("SqliteVectorStore: prepare insert failed");
    }
    SqliteStmt stmt(raw_stmt);

    SqliteTransaction txn(impl_->db);
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& id : ids)
    {
        const Document* doc = nullptr;
        for (auto it = docs_.rbegin(); it != docs_.rend(); ++it)
        {
            if (it->id == id)
            {
                doc = &*it;
                break;
            }
        }
        if (!doc)
        {
            continue;
        }

        json meta = json::object();
        for (const auto& kv : doc->metadata)
        {
            meta[kv.first] = kv.second;
        }
        std::string meta_str = meta.dump();

        sqlite3_bind_text(stmt.get(), 1, doc->id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt.get(), 2, doc->content.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt.get(), 3, meta_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(stmt.get(), 4, doc->embedding.data(),
                          static_cast<int>(doc->embedding.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        if (sqlite3_step(stmt.get()) != SQLITE_DONE)
        {
            LOG_WARN("[SqliteVectorStore] insert failed for id {}", doc->id);
        }
        sqlite3_reset(stmt.get());
    }
    txn.commit();
    return ids;
}

std::size_t SqliteVectorStore::delete_documents(const std::vector<std::string>& ids)
{
    std::size_t removed = InMemoryVectorStore::delete_documents(ids);
    for (const auto& id : ids)
    {
        persist_delete_(id);
    }
    return removed;
}

void SqliteVectorStore::persist_delete_(const std::string& id)
{
    std::string sql = "DELETE FROM documents WHERE id = ?;";
    sqlite3_stmt* raw_stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw_stmt, nullptr) == SQLITE_OK)
    {
        SqliteStmt stmt(raw_stmt);
        sqlite3_bind_text(stmt.get(), 1, id.c_str(), -1, SQLITE_STATIC);
        sqlite3_step(stmt.get());
    }
}

void SqliteVectorStore::update_document(const std::string& id, Document doc)
{
    InMemoryVectorStore::update_document(id, std::move(doc));
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& d : docs_)
    {
        if (d.id == id)
        {
            persist_update_(d);
            return;
        }
    }
}

void SqliteVectorStore::persist_update_(const Document& doc)
{
    std::string sql =
        "INSERT OR REPLACE INTO documents(id, content, metadata, embedding)"
        " VALUES(?,?,?,?);";
    sqlite3_stmt* raw_stmt = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw_stmt, nullptr) != SQLITE_OK)
    {
        return;
    }
    SqliteStmt stmt(raw_stmt);
    json meta = json::object();
    for (const auto& kv : doc.metadata)
    {
        meta[kv.first] = kv.second;
    }
    sqlite3_bind_text(stmt.get(), 1, doc.id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt.get(), 2, doc.content.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt.get(), 3, meta.dump().c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt.get(), 4, doc.embedding.data(),
                      static_cast<int>(doc.embedding.size() * sizeof(float)),
                      SQLITE_STATIC);
    sqlite3_step(stmt.get());
}

// ---------------- FaissVectorStore ----------------

#ifdef LC_HAS_FAISS

namespace
{

class FaissIndexHolder
{
public:
    explicit FaissIndexHolder(faiss::Index* idx)
        : idx_(idx)
    {
    }
    ~FaissIndexHolder()
    {
        delete idx_;
    }
    FaissIndexHolder(const FaissIndexHolder&) = delete;
    FaissIndexHolder& operator=(const FaissIndexHolder&) = delete;

    faiss::Index* get() const
    {
        return idx_;
    }

private:
    faiss::Index* idx_;
};

} // namespace

struct FaissVectorStore::Impl
{
    std::unique_ptr<FaissIndexHolder> index;
    int dim = 0;
    std::size_t count = 0;
    int nlist = 0;
    int nprobe = 1;
};

FaissVectorStore::FaissVectorStore(embedding::EmbeddingPtr embedder,
                                   int nlist,
                                   int nprobe)
    : IVectorStore(),
      embedder_(std::move(embedder)),
      impl_(std::make_unique<Impl>())
{
    if (!embedder_)
    {
        throw LCError("FaissVectorStore: embedder is null");
    }
    impl_->dim = embedder_->dimension();
    impl_->nlist = nlist;
    impl_->nprobe = nprobe;

    faiss::Index* idx = nullptr;
    if (nlist > 0)
    {
        auto* quantizer = new faiss::IndexFlatIP(impl_->dim);
        idx = new faiss::IndexIVFFlat(quantizer, impl_->dim, nlist);
    }
    else
    {
        idx = new faiss::IndexFlatIP(impl_->dim);
    }
    impl_->index = std::make_unique<FaissIndexHolder>(idx);
}

FaissVectorStore::~FaissVectorStore() = default;

std::vector<std::string> FaissVectorStore::add_documents(std::vector<Document> docs)
{
    if (docs.empty())
    {
        return {};
    }

    std::vector<std::string> texts;
    texts.reserve(docs.size());
    for (auto& d : docs)
    {
        if (d.id.empty())
        {
            d.id = gen_id();
        }
        texts.push_back(d.content);
    }

    auto vecs = embedder_->embed_documents(texts);
    std::vector<float> flat;
    flat.reserve(static_cast<std::size_t>(vecs.size()) * impl_->dim);
    for (const auto& v : vecs)
    {
        flat.insert(flat.end(), v.begin(), v.end());
    }

    auto* idx = impl_->index->get();
    if (dynamic_cast<faiss::IndexIVFFlat*>(idx) != nullptr &&
        idx->ntotal == 0 && impl_->nlist > 0)
    {
        idx->train(static_cast<faiss::idx_t>(vecs.size()), flat.data());
    }
    idx->add(static_cast<faiss::idx_t>(vecs.size()), flat.data());

    std::vector<std::string> ids;
    ids.reserve(docs.size());

    std::lock_guard<std::mutex> lk(mu_);
    for (std::size_t i = 0; i < docs.size(); ++i)
    {
        docs[i].embedding = std::move(vecs[i]);
        ids.push_back(docs[i].id);
        docs_.push_back(std::move(docs[i]));
    }
    impl_->count += docs.size();
    return ids;
}

std::vector<ScoredDocument> FaissVectorStore::similarity_search(
    const std::string& query, int k,
    const std::unordered_map<std::string, std::string>& filter)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (docs_.empty() || k <= 0)
    {
        return {};
    }

    auto q = embedder_->embed_query(query);
    if (q.size() != static_cast<std::size_t>(impl_->dim))
    {
        throw LCError("FaissVectorStore: query dimension mismatch");
    }

    std::vector<std::size_t> filtered_indices;
    if (!filter.empty())
    {
        for (std::size_t i = 0; i < docs_.size(); ++i)
        {
            bool match = true;
            for (const auto& kv : filter)
            {
                auto it = docs_[i].metadata.find(kv.first);
                if (it == docs_[i].metadata.end() || it->second != kv.second)
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                filtered_indices.push_back(i);
            }
        }
        if (filtered_indices.empty())
        {
            return {};
        }
    }

    auto* idx = impl_->index->get();
    if (auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(idx))
    {
        ivf->nprobe = impl_->nprobe;
    }

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    idx->search(1, q.data(), k, distances.data(), labels.data());

    std::vector<ScoredDocument> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i)
    {
        if (labels[i] < 0)
        {
            continue;
        }
        std::size_t doc_idx = static_cast<std::size_t>(labels[i]);
        if (doc_idx >= docs_.size())
        {
            continue;
        }
        if (!filter.empty() &&
            std::find(filtered_indices.begin(), filtered_indices.end(), doc_idx) ==
                filtered_indices.end())
        {
            continue;
        }
        out.push_back(ScoredDocument{docs_[doc_idx], distances[i]});
    }
    return out;
}

std::size_t FaissVectorStore::delete_documents(const std::vector<std::string>& ids)
{
    std::lock_guard<std::mutex> lk(mu_);
    std::size_t removed = 0;
    for (const auto& id : ids)
    {
        auto it = std::remove_if(docs_.begin(), docs_.end(),
                                 [&id](const Document& d) { return d.id == id; });
        if (it != docs_.end())
        {
            removed += static_cast<std::size_t>(std::distance(it, docs_.end()));
            docs_.erase(it, docs_.end());
        }
    }
    impl_->count = docs_.size();
    return removed;
}

void FaissVectorStore::update_document(const std::string& id, Document doc)
{
    (void)id;
    (void)doc;
    throw LCError("FaissVectorStore: update_document not yet implemented");
}

std::size_t FaissVectorStore::size() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return impl_->count;
}

#else

// Stub implementations when FAISS is not available.
// Define a minimal Impl so unique_ptr works.
struct FaissVectorStore::Impl {};

FaissVectorStore::FaissVectorStore(embedding::EmbeddingPtr, int, int)
    : impl_(std::make_unique<Impl>())
{
    throw LCError("FaissVectorStore: FAISS support not compiled in. Rebuild with -DLC_ENABLE_FAISS=ON");
}

FaissVectorStore::~FaissVectorStore() = default;

std::vector<std::string> FaissVectorStore::add_documents(std::vector<Document>)
{
    return {};
}

std::vector<ScoredDocument> FaissVectorStore::similarity_search(
    const std::string&, int,
    const std::unordered_map<std::string, std::string>&)
{
    return {};
}

std::size_t FaissVectorStore::delete_documents(const std::vector<std::string>&)
{
    return 0;
}

void FaissVectorStore::update_document(const std::string&, Document)
{
}

std::size_t FaissVectorStore::size() const
{
    return 0;
}

#endif // LC_HAS_FAISS

} // namespace vectorstore
} // namespace langchain
