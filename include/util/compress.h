// langchain/util/compress.h
// Thin wrapper around bundled zlib for in-memory deflate / inflate. Provides
// both raw zlib stream and gzip framing. Use for caching large embeddings,
// transport compression, etc.
#pragma once

#include <cstdint>
#include <string>

namespace langchain
{
namespace util
{
namespace compress
{

// Returns the zlib runtime version (e.g. "1.3.2"). Named with `runtime_`
// prefix to avoid the `#define zlib_version zlibVersion()` macro in zlib.h.
const char* zlib_runtime_version();

// Raw zlib (RFC 1950) deflate. `level` is 0..9 or -1 for default.
// Named with `_str` suffix to avoid collision with zlib's C functions
// of the same names which are visible at global scope after <zlib.h>.
std::string deflate_str(const std::string& src, int level = -1);
std::string inflate_str(const std::string& src);

// Gzip-wrapped (RFC 1952). Interoperable with `gzip` / `gunzip` CLI tools.
std::string gzip(const std::string& src, int level = -1);
std::string gunzip(const std::string& src);

} // namespace compress
} // namespace util
} // namespace langchain
