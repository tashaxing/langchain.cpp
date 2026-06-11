// langchain/util/fs.h
// Minimal cross-platform filesystem helpers. Uses std::filesystem (C++17);
// on GCC <9 the CMake build links -lstdc++fs so this works on RHEL 8.6 GCC 8.5.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace langchain
{
namespace util
{
namespace fs
{

bool exists(const std::string& path);
bool is_file(const std::string& path);
bool is_dir(const std::string& path);

// Bytes; -1 if file does not exist.
std::int64_t file_size(const std::string& path);

// Whole-file read; throws LCError on failure. Binary safe.
std::string read_file(const std::string& path);

// Writes content, creating parent dirs as needed. Throws on failure.
void write_file(const std::string& path,
                const std::string& content,
                bool create_dirs = true);

// `mkdir -p` semantics. Returns true if the path now exists as a directory.
bool make_dirs(const std::string& path);

// Delete a file or empty directory. Returns true on success or if missing.
bool remove(const std::string& path);

// Non-recursive directory listing. Returns the entry names (not full paths).
std::vector<std::string> list_dir(const std::string& path);

// Cross-platform path join (uses '/').
std::string join(const std::string& a, const std::string& b);

// Returns the directory portion of `path` (or "" if none).
std::string parent(const std::string& path);

// Returns the filename portion of `path`.
std::string filename(const std::string& path);

// Returns the extension including the leading '.', or "" if none.
std::string extension(const std::string& path);

} // namespace fs
} // namespace util
} // namespace langchain
