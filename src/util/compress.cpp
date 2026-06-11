// src/util/compress.cpp — zlib-backed in-memory (de)compression.
#include "util/compress.h"
#include "util/common.h"

#include <zlib.h>

#include <cstring>
#include <vector>

namespace langchain
{
namespace util
{
namespace compress
{

namespace
{

// windowBits magic per zlib docs:
//   8..15  = zlib stream
//   -8..-15 = raw deflate (no header)
//   16+(8..15) = gzip stream
constexpr int kZlibWindow = 15;
constexpr int kGzipWindow = 15 + 16;

std::string deflate_with_window(const std::string& src, int level, int window_bits)
{
    z_stream zs{};
    if (deflateInit2(&zs, level, Z_DEFLATED, window_bits, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    {
        throw LCError("compress::deflate: deflateInit2 failed");
    }

    zs.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(src.data()));
    zs.avail_in = static_cast<uInt>(src.size());

    std::string out;
    constexpr std::size_t kChunk = 32 * 1024;
    std::vector<char> buf(kChunk);

    int ret;
    do
    {
        zs.next_out  = reinterpret_cast<Bytef*>(buf.data());
        zs.avail_out = static_cast<uInt>(buf.size());
        ret = ::deflate(&zs, Z_FINISH);
        if (ret == Z_STREAM_ERROR)
        {
            deflateEnd(&zs);
            throw LCError("compress::deflate: stream error");
        }
        out.append(buf.data(), buf.size() - zs.avail_out);
    } while (zs.avail_out == 0);

    deflateEnd(&zs);
    return out;
}

std::string inflate_with_window(const std::string& src, int window_bits)
{
    z_stream zs{};
    if (inflateInit2(&zs, window_bits) != Z_OK)
    {
        throw LCError("compress::inflate: inflateInit2 failed");
    }

    zs.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(src.data()));
    zs.avail_in = static_cast<uInt>(src.size());

    std::string out;
    constexpr std::size_t kChunk = 32 * 1024;
    std::vector<char> buf(kChunk);

    int ret;
    do
    {
        zs.next_out  = reinterpret_cast<Bytef*>(buf.data());
        zs.avail_out = static_cast<uInt>(buf.size());
        ret = ::inflate(&zs, Z_NO_FLUSH);
        switch (ret)
        {
            case Z_NEED_DICT:
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                inflateEnd(&zs);
                throw LCError(std::string("compress::inflate: error ") +
                              std::to_string(ret));
            default: break;
        }
        out.append(buf.data(), buf.size() - zs.avail_out);
    } while (ret != Z_STREAM_END && zs.avail_in > 0);

    inflateEnd(&zs);
    return out;
}

} // namespace

const char* zlib_runtime_version()
{
    return zlibVersion();
}

std::string deflate_str(const std::string& src, int level)
{
    return deflate_with_window(src, level, kZlibWindow);
}

std::string inflate_str(const std::string& src)
{
    return inflate_with_window(src, kZlibWindow);
}

std::string gzip(const std::string& src, int level)
{
    return deflate_with_window(src, level, kGzipWindow);
}

std::string gunzip(const std::string& src)
{
    return inflate_with_window(src, kGzipWindow);
}

} // namespace compress
} // namespace util
} // namespace langchain
