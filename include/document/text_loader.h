// langchain/document/text_loader.h
// Concrete loaders for common file types.
#pragma once

#include "document/base_loader.h"

#include <string>

namespace langchain
{
namespace document
{

// ---------------------------------------------------------------------------
// TextFileLoader -- load a plain text file as a single Document.
// ---------------------------------------------------------------------------
class TextFileLoader : public BaseLoader
{
public:
    explicit TextFileLoader(std::string path);

    std::vector<Document> load() override;

private:
    std::string path_;
};

} // namespace document
} // namespace langchain
