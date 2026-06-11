// app_hooks.h -- Hook registration for observability.
#pragma once

namespace smart_app
{

// Register lifecycle hooks that log via the application logger.
void register_hooks();

} // namespace smart_app
