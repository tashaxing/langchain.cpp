// app_hooks.cpp -- Hook registration for observability.
#include "app_hooks.h"

#include "hook/hook.h"
#include "util/logging.h"

#include <sstream>

namespace smart_app
{

void register_hooks()
{
    auto& hooks = langchain::hook::HookManager::global();

    hooks.add("app-logger", [](langchain::hook::HookContext& ctx)
    {
        std::ostringstream oss;
        oss << "[" << langchain::hook::to_string(ctx.phase) << "]"
            << " component=" << ctx.component
            << " call_id=" << ctx.call_id;

        if (ctx.elapsed.count() > 0)
        {
            oss << " elapsed_us=" << ctx.elapsed.count();
        }
        if (ctx.agent_input)
        {
            oss << " agent_input=\"" << *ctx.agent_input << "\"";
        }
        if (ctx.tool_name)
        {
            oss << " tool=" << *ctx.tool_name;
        }

        switch (ctx.phase)
        {
        case langchain::hook::Phase::BeforeLLM:
        case langchain::hook::Phase::BeforeApi:
            LOG_INFO("{}", oss.str());
            break;
        case langchain::hook::Phase::AfterLLM:
        case langchain::hook::Phase::AfterApi:
            LOG_INFO("{}", oss.str());
            break;
        case langchain::hook::Phase::BeforeTool:
            LOG_DEBUG("{}", oss.str());
            break;
        case langchain::hook::Phase::AfterTool:
            LOG_DEBUG("{}", oss.str());
            break;
        default:
            LOG_DEBUG("{}", oss.str());
            break;
        }
    });

    LOG_INFO("Hooks registered: app-logger");
}

} // namespace smart_app
