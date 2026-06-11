// app_agent.h -- Agent factory.
#pragma once

#include "langchain.h"

#include <memory>

namespace smart_app
{

// Build a ToolCallingAgent with tools, skills, and memory.
std::shared_ptr<langchain::agent::ToolCallingAgent> build_agent(
    langchain::llm::LLMPtr llm,
    langchain::tool::ToolRegistry tools,
    langchain::skill::SkillRegistry skills,
    langchain::memory::MemoryPtr mem);

} // namespace smart_app
