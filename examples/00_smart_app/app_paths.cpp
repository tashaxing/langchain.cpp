// app_paths.cpp -- Cross-platform executable path resolution.
#include "app_paths.h"

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace smart_app
{

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

std::string get_executable_dir()
{
    std::vector<wchar_t> buffer(4096);
    DWORD len = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (len == 0 || len >= buffer.size())
    {
        return "";
    }
    std::wstring wpath(buffer.data(), len);
    fs::path p(wpath);
    return p.parent_path().string();
}

#elif defined(__APPLE__)

#include <mach-o/dyld.h>

std::string get_executable_dir()
{
    std::vector<char> buffer(4096);
    uint32_t size = static_cast<uint32_t>(buffer.size());
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    {
        buffer.resize(size);
        if (_NSGetExecutablePath(buffer.data(), &size) != 0)
        {
            return "";
        }
    }
    fs::path p(buffer.data());
    return p.parent_path().string();
}

#else // Linux

std::string get_executable_dir()
{
    fs::path p = fs::read_symlink("/proc/self/exe");
    return p.parent_path().string();
}

#endif

std::string get_app_base_dir()
{
    std::string exe_dir = get_executable_dir();
    if (exe_dir.empty())
    {
        return "";
    }

    fs::path p(exe_dir);
    // If executable is inside bin/, return the parent directory.
    if (p.filename() == "bin" || p.filename() == "Bin")
    {
        return p.parent_path().string();
    }
    return exe_dir;
}

std::string resolve_app_path(const std::string& path)
{
    if (path.empty())
    {
        return path;
    }

    fs::path p(path);
    // If already absolute, return as-is.
    if (p.is_absolute())
    {
        return path;
    }

    std::string base = get_app_base_dir();
    if (base.empty())
    {
        return path; // Fallback: return relative path unchanged.
    }

    return (fs::path(base) / p).string();
}

} // namespace smart_app
