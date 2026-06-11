// src/document/text_loader.cpp -- document loader implementations.
#include "document/text_loader.h"

#include "util/fs.h"

#include <fstream>
#include <sstream>

namespace langchain
{
namespace document
{

// ---------------------------------------------------------------------------
// BaseLoader
// ---------------------------------------------------------------------------
void BaseLoader::lazy_load(const std::function<void(Document)>& cb)
{
    auto docs = load();
    for (auto& doc : docs)
    {
        cb(std::move(doc));
    }
}

// ---------------------------------------------------------------------------
// TextFileLoader
// ---------------------------------------------------------------------------
TextFileLoader::TextFileLoader(std::string path)
    : path_(std::move(path))
{
}

std::vector<Document> TextFileLoader::load()
{
    std::ifstream ifs(path_);
    if (!ifs.is_open())
    {
        throw LCError("TextFileLoader: cannot open file: " + path_);
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();

    Document doc;
    doc.id      = path_;
    doc.content = oss.str();
    doc.metadata["source"] = path_;
    return { std::move(doc) };
}

} // namespace document
} // namespace langchain
