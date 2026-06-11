// app_tools.h -- Tool registry builder.
#pragma once

#include "langchain.h"

namespace smart_app
{

// Build a tool registry with built-in and custom tools based on config.
langchain::tool::ToolRegistry build_tool_registry();

} // namespace smart_app
