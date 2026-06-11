// src/util/common.cpp — Role <-> string helpers + multimodal file loader.
#include "util/common.h"
#include "util/fs.h"

#include <sstream>

namespace langchain
{

namespace
{

std::string mime_from_extension(const std::string& ext)
{
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif")  return "image/gif";
    if (ext == ".webp") return "image/webp";
    if (ext == ".bmp")  return "image/bmp";
    if (ext == ".svg")  return "image/svg+xml";
    return "image/png"; // default
}

// Base64 encode binary data.
std::string base64_encode(const std::string& data)
{
    static const char* chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= data.size())
    {
        std::uint32_t b =
            (static_cast<unsigned char>(data[i]) << 16) |
            (static_cast<unsigned char>(data[i + 1]) << 8) |
            static_cast<unsigned char>(data[i + 2]);
        out.push_back(chars[(b >> 18) & 0x3F]);
        out.push_back(chars[(b >> 12) & 0x3F]);
        out.push_back(chars[(b >> 6) & 0x3F]);
        out.push_back(chars[b & 0x3F]);
        i += 3;
    }
    if (i + 1 == data.size())
    {
        std::uint32_t b = static_cast<unsigned char>(data[i]) << 16;
        out.push_back(chars[(b >> 18) & 0x3F]);
        out.push_back(chars[(b >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    }
    else if (i + 2 == data.size())
    {
        std::uint32_t b =
            (static_cast<unsigned char>(data[i]) << 16) |
            (static_cast<unsigned char>(data[i + 1]) << 8);
        out.push_back(chars[(b >> 18) & 0x3F]);
        out.push_back(chars[(b >> 12) & 0x3F]);
        out.push_back(chars[(b >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

} // namespace

const char* to_string(Role r)
{
    switch (r)
    {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
        case Role::Tool:      return "tool";
    }
    return "user";
}

Role role_from_string(const std::string& s)
{
    if (s == "system")    return Role::System;
    if (s == "assistant") return Role::Assistant;
    if (s == "tool")      return Role::Tool;
    return Role::User;
}

Message Message::user_with_image_file(std::string text,
                                      const std::string& file_path,
                                      std::string mime_type)
{
    auto bytes = util::fs::read_file(file_path);
    if (mime_type.empty())
    {
        mime_type = mime_from_extension(util::fs::extension(file_path));
    }
    return user_with_image_base64(std::move(text), base64_encode(bytes), std::move(mime_type));
}

} // namespace langchain
