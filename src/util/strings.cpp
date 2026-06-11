// src/util/strings.cpp — string helpers used by prompt/agent parsing.
#include "util/strings.h"

#include <algorithm>
#include <cctype>

namespace langchain
{
namespace strings
{

std::string ltrim(std::string s)
{
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         [](unsigned char c) { return !std::isspace(c); }));
    return s;
}

std::string rtrim(std::string s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         [](unsigned char c) { return !std::isspace(c); }).base(),
            s.end());
    return s;
}

std::string trim(std::string s)
{
    return rtrim(ltrim(std::move(s)));
}

bool starts_with(const std::string& s, const std::string& p)
{
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

bool contains(const std::string& s, const std::string& p)
{
    return s.find(p) != std::string::npos;
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i)
    {
        if (i == s.size() || s[i] == sep)
        {
            out.emplace_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to)
{
    if (from.empty())
    {
        return s;
    }
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos)
    {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

} // namespace strings
} // namespace langchain
