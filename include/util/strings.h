// langchain/util/strings.h
// Small string helpers used by prompt/agent parsing.
#pragma once

#include <string>
#include <vector>

namespace langchain
{
namespace strings
{

std::string ltrim(std::string s);
std::string rtrim(std::string s);
std::string trim(std::string s);

bool starts_with(const std::string& s, const std::string& p);
bool contains(const std::string& s, const std::string& p);

std::vector<std::string> split(const std::string& s, char sep);

std::string replace_all(std::string s, const std::string& from, const std::string& to);

} // namespace strings
} // namespace langchain
