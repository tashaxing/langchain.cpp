// src/util/fs.cpp
#include "util/fs.h"
#include "util/common.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <system_error>

// Use std::filesystem; on GCC<9 the library is linked via stdc++fs from CMake.
#if defined(__GNUC__) && (__GNUC__ < 9) && !defined(__clang__)
#  include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#else
#  include <filesystem>
namespace stdfs = std::filesystem;
#endif

namespace langchain
{
namespace util
{
namespace fs
{

bool exists(const std::string& path)
{
    std::error_code ec;
    return stdfs::exists(path, ec);
}

bool is_file(const std::string& path)
{
    std::error_code ec;
    return stdfs::is_regular_file(path, ec);
}

bool is_dir(const std::string& path)
{
    std::error_code ec;
    return stdfs::is_directory(path, ec);
}

std::int64_t file_size(const std::string& path)
{
    std::error_code ec;
    auto sz = stdfs::file_size(path, ec);
    if (ec)
    {
        return -1;
    }
    return static_cast<std::int64_t>(sz);
}

std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw LCError("fs::read_file: cannot open '" + path + "': " +
                      std::strerror(errno));
    }
    in.seekg(0, std::ios::end);
    auto end = in.tellg();
    in.seekg(0, std::ios::beg);
    std::string out;
    if (end > 0)
    {
        out.resize(static_cast<std::size_t>(end));
        in.read(&out[0], end);
    }
    return out;
}

void write_file(const std::string& path,
                const std::string& content,
                bool create_dirs)
{
    if (create_dirs)
    {
        auto p = parent(path);
        if (!p.empty())
        {
            make_dirs(p);
        }
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw LCError("fs::write_file: cannot open '" + path + "': " +
                      std::strerror(errno));
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!out)
    {
        throw LCError("fs::write_file: write failed for '" + path + "'");
    }
}

bool make_dirs(const std::string& path)
{
    std::error_code ec;
    if (stdfs::exists(path, ec))
    {
        return stdfs::is_directory(path, ec);
    }
    stdfs::create_directories(path, ec);
    return !ec;
}

bool remove(const std::string& path)
{
    std::error_code ec;
    if (!stdfs::exists(path, ec))
    {
        return true;
    }
    return stdfs::remove(path, ec);
}

std::vector<std::string> list_dir(const std::string& path)
{
    std::vector<std::string> out;
    std::error_code ec;
    stdfs::directory_iterator it(path, ec), end;
    if (ec)
    {
        return out;
    }
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }
        out.push_back(it->path().filename().string());
    }
    return out;
}

std::string join(const std::string& a, const std::string& b)
{
    if (a.empty())
    {
        return b;
    }
    if (b.empty())
    {
        return a;
    }
    char last = a.back();
    if (last == '/' || last == '\\')
    {
        return a + b;
    }
    return a + "/" + b;
}

std::string parent(const std::string& path)
{
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return std::string();
    }
    return path.substr(0, pos);
}

std::string filename(const std::string& path)
{
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return path;
    }
    return path.substr(pos + 1);
}

std::string extension(const std::string& path)
{
    auto name = filename(path);
    auto pos  = name.find_last_of('.');
    if (pos == std::string::npos || pos == 0)
    {
        return std::string();
    }
    return name.substr(pos);
}

} // namespace fs
} // namespace util
} // namespace langchain
