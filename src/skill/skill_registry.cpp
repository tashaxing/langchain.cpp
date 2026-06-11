// src/skill/skill_registry.cpp -- SkillRegistry implementation.
#include "skill/skill_registry.h"

namespace langchain
{
namespace skill
{

void SkillRegistry::add(SkillPtr skill)
{
    if (!skill)
    {
        return;
    }
    skills_[skill->name()] = std::move(skill);
}

SkillPtr SkillRegistry::get(const std::string& name) const
{
    auto it = skills_.find(name);
    return it == skills_.end() ? nullptr : it->second;
}

std::vector<std::string> SkillRegistry::names() const
{
    std::vector<std::string> out;
    out.reserve(skills_.size());
    for (const auto& kv : skills_)
    {
        out.push_back(kv.first);
    }
    return out;
}

bool SkillRegistry::empty() const
{
    return skills_.empty();
}

std::size_t SkillRegistry::size() const
{
    return skills_.size();
}

tool::ToolRegistry SkillRegistry::as_tools() const
{
    tool::ToolRegistry out;
    for (const auto& kv : skills_)
    {
        if (auto t = skill_to_tool(kv.second))
        {
            out.add(std::move(t));
        }
    }
    return out;
}

} // namespace skill
} // namespace langchain
