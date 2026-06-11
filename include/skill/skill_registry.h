// langchain/skill/skill_registry.h
// SkillRegistry -- a container for ISkill instances, analogous to
// tool::ToolRegistry. Skills can be exported as tools so that agents
// can invoke them.
#pragma once

#include "skill/skill.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace skill
{

class SkillRegistry
{
public:
    void add(SkillPtr skill);

    SkillPtr get(const std::string& name) const;

    std::vector<std::string> names() const;

    bool empty() const;

    std::size_t size() const;

    // Export every registered skill as a tool::FunctionTool.
    // This lets an agent use skills the same way it uses tools.
    tool::ToolRegistry as_tools() const;

private:
    std::map<std::string, SkillPtr> skills_;
};

} // namespace skill
} // namespace langchain
