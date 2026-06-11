// langchain/document/base_loader.h
// Base interface for document loaders.
#pragma once

#include "util/common.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace document
{

// ---------------------------------------------------------------------------
// BaseLoader -- abstract interface for document loaders.
// ---------------------------------------------------------------------------
class BaseLoader
{
public:
    virtual ~BaseLoader() = default;

    // Load all documents from this source.
    virtual std::vector<Document> load() = 0;

    // Load lazily (one Document at a time). Default calls load() and iterates.
    virtual void lazy_load(const std::function<void(Document)>& cb);
};

using LoaderPtr = std::shared_ptr<BaseLoader>;

} // namespace document
} // namespace langchain
