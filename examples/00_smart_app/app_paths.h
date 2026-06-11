// app_paths.h -- Cross-platform path helpers for 00_smart_app.
// Provides the application's base directory (where bin/, config/, log/ live)
// resolved from the executable's location.
#pragma once

#include <string>

namespace smart_app
{

// Return the directory containing the current executable.
// On Linux: reads /proc/self/exe
// On macOS: uses _NSGetExecutablePath
// On Windows: uses GetModuleFileNameW
std::string get_executable_dir();

// Return the application base directory (parent of bin/).
// If the executable is in a "bin" subdirectory, returns the parent.
// Otherwise returns the executable directory.
std::string get_app_base_dir();

// Resolve a relative path against the application base directory.
// If `path` is already absolute, returns it unchanged.
std::string resolve_app_path(const std::string& path);

} // namespace smart_app
