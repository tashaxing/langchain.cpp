#include "prompt/prompt_template.h"

#include "util/strings.h"

#include <regex>
#include <unordered_set>

namespace langchain
{
namespace prompt
{

namespace
{

// Single-brace placeholders {var} matching identifiers only — keeps the regex
// simple and avoids tangling with JSON braces in the surrounding text.
const std::regex& placeholder_re()
{
    static const std::regex re(R"(\{([A-Za-z_][A-Za-z0-9_]*)\})");
    return re;
}

} // namespace

PromptTemplate::PromptTemplate(std::string tmpl)
    : template_(std::move(tmpl))
{
    std::unordered_set<std::string> seen;
    auto begin = std::sregex_iterator(template_.begin(), template_.end(), placeholder_re());
    auto end   = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
    {
        std::string name = (*it)[1].str();
        if (seen.insert(name).second)
        {
            input_variables_.push_back(name);
        }
    }
}

std::string PromptTemplate::format(
    const std::unordered_map<std::string, std::string>& vars,
    bool allow_missing) const
{
    std::string out;
    out.reserve(template_.size());
    auto begin = std::sregex_iterator(template_.begin(), template_.end(), placeholder_re());
    auto end   = std::sregex_iterator();
    std::size_t last = 0;
    for (auto it = begin; it != end; ++it)
    {
        const auto& m = *it;
        std::size_t pos = static_cast<std::size_t>(m.position(0));
        out.append(template_, last, pos - last);
        std::string key = m[1].str();
        auto found = vars.find(key);
        if (found == vars.end())
        {
            if (!allow_missing)
            {
                throw LCError("PromptTemplate: missing variable '" + key + "'");
            }
            out.append(m[0].str());
        }
        else
        {
            out.append(found->second);
        }
        last = pos + static_cast<std::size_t>(m.length(0));
    }
    out.append(template_, last, std::string::npos);
    return out;
}

std::vector<Message> ChatPromptTemplate::format_messages(
    const std::unordered_map<std::string, std::string>& vars) const
{
    std::vector<Message> msgs;
    msgs.reserve(slots_.size());
    for (const auto& s : slots_)
    {
        msgs.emplace_back(s.role, s.tmpl.format(vars));
    }
    return msgs;
}

} // namespace prompt
} // namespace langchain
