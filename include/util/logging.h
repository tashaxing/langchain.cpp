// langchain/util/logging.h
// Thin wrapper around spdlog with configurable file rotation, level, and
// rich log-line formatting (timestamp + thread-id + level + file:line + msg).
//
// Usage:
//   langchain::log::init("logs", "app.log", spdlog::level::info, 500 * 1024 * 1024);
//   LC_INFO("Hello {}", "world");
//
// The rotating sink keeps the *newest* slice at index 0 (app.log) and pushes
// older slices to higher indices (app.1.log, app.2.log, ...).
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/dist_sink.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace langchain
{
namespace log
{

// ---------------------------------------------------------------------------
// Init / reconfigure the global logger.
//
//   log_dir        : directory for log files (created if missing).
//   log_filename   : base file name, e.g. "app.log".
//   level          : spdlog::level::info / debug / warn / error.
//   max_file_size  : rotation threshold in bytes (default 500 MiB).
//   max_files      : number of rotated slices to keep (default 10).
//   console        : also echo to stderr (default true).
//
// Safe to call multiple times — replaces the existing logger.
// ---------------------------------------------------------------------------
void init(const std::string& log_dir,
          const std::string& log_filename,
          spdlog::level::level_enum level = spdlog::level::info,
          std::size_t max_file_size = 500 * 1024 * 1024,
          std::size_t max_files = 10,
          bool console = true);

// Change log level at runtime.
void set_level(spdlog::level::level_enum level);

// ---------------------------------------------------------------------------
// Internal: fetch the current logger (falls back to stderr on error).
// ---------------------------------------------------------------------------
spdlog::logger& get();

} // namespace log
} // namespace langchain

// ---------------------------------------------------------------------------
// Public macros — capture __FILE__ / __LINE__ automatically.
// ---------------------------------------------------------------------------
#define LC_LOG(level, ...) \
    langchain::log::get().log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)

#define LOG_TRACE(...) LC_LOG(spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...) LC_LOG(spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...)  LC_LOG(spdlog::level::info,  __VA_ARGS__)
#define LOG_WARN(...)  LC_LOG(spdlog::level::warn,  __VA_ARGS__)
#define LOG_ERROR(...) LC_LOG(spdlog::level::err,   __VA_ARGS__)
#define LOG_FATAL(...) LC_LOG(spdlog::level::critical, __VA_ARGS__)
