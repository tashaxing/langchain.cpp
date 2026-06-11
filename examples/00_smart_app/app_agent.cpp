// app_agent.cpp -- Agent factory.
#include "app_agent.h"
#include "app_config.h"

#include "util/logging.h"

namespace smart_app
{

std::shared_ptr<langchain::agent::ToolCallingAgent> build_agent(
    langchain::llm::LLMPtr llm,
    langchain::tool::ToolRegistry tools,
    langchain::skill::SkillRegistry skills,
    langchain::memory::MemoryPtr mem)
{
    using namespace langchain;

    langchain::agent::AgentConfig acfg = get_agent_config();

    auto agent = std::make_shared<agent::ToolCallingAgent>(
        std::move(llm),
        std::move(tools),
        std::move(skills),
        acfg,
        std::move(mem));

    LOG_INFO("Agent created: type=tool_calling max_iterations={}", acfg.max_iterations);
    return agent;
}

} // namespace smart_app
