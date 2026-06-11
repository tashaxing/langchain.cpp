// main.cpp -- Entry point for 00_smart_app.
//
// A comprehensive AI smart agent application built on langchain.cpp.
// Usage:
//   00_smart_app [--config <path>]
//
// The app starts an OpenAI-compatible HTTP server with:
//   - Multiple LLM models registered
//   - ToolCallingAgent with tools, skills, and persistent memory
//   - Optional RAG pipeline with SQLite vector store
//   - Custom routes for memory inspection, config reload, and document ingestion
//   - Streaming and non-streaming chat completions
//   - Multimodal input pass-through
//
#include "app_config.h"
#include "app_paths.h"
#include "app_server.h"
#include "util/logging.h"
#include "util/timer.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace
{
std::atomic<bool> g_stop{false};

void on_signal(int)
{
    g_stop.store(true);
}

std::string parse_config_path(int argc, char* argv[])
{
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "--config")
        {
            return argv[i + 1];
        }
    }
    // Default: resolve relative to app base dir (where bin/, config/ live).
    return smart_app::resolve_app_path("config/app_config.xml");
}

spdlog::level::level_enum parse_log_level(const std::string& s)
{
    std::string lower;
    lower.reserve(s.size());
    for (unsigned char c : s)
    {
        lower += static_cast<char>(std::tolower(c));
    }
    if (lower == "trace") return spdlog::level::trace;
    if (lower == "debug") return spdlog::level::debug;
    if (lower == "info")  return spdlog::level::info;
    if (lower == "warn" || lower == "warning") return spdlog::level::warn;
    if (lower == "error" || lower == "err") return spdlog::level::err;
    if (lower == "fatal" || lower == "critical") return spdlog::level::critical;
    return spdlog::level::info;
}

} // namespace

int main(int argc, char* argv[])
{
    // ---- Initialize app base dir (must be first) ----
    smart_app::init_app_base_dir();

    std::string config_path = parse_config_path(argc, argv);

    // ---- Load configuration ----
    if (!smart_app::load_app_config(config_path))
    {
        // Logging not yet initialized, use stderr as fallback.
        std::cerr << "Failed to load configuration. Exiting.\n";
        return 1;
    }

    // ---- Initialize logging ----
    auto log_cfg = smart_app::get_log_config();
    // Generate log filename with startup timestamp: smart_app_YYYYMMDD_HHMMSS.log
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream log_name_oss;
    log_name_oss << "smart_app_"
                 << std::put_time(&tm, "%Y%m%d_%H%M%S")
                 << ".log";
    std::string log_filename = log_name_oss.str();
    langchain::log::init(
        log_cfg.dir,
        log_filename,
        parse_log_level(log_cfg.level),
        log_cfg.max_size,
        log_cfg.max_files,
        log_cfg.console);

    LOG_INFO("00_smart_app starting...");
    LOG_INFO("Config loaded from: {}", config_path);

    // ---- Signal handling ----
    std::signal(SIGINT, on_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, on_signal);
#endif

    // ---- Build and start server ----
    smart_app::AppServer app;
    try
    {
        app.build();
    }
    catch (const std::exception& e)
    {
        LOG_FATAL("Server build failed: {}", e.what());
        return 1;
    }

    app.start();

    // ---- Heartbeat timer ----
    langchain::util::Timer heartbeat_timer;
    heartbeat_timer.start_interval(
        std::chrono::seconds(30),
        [] { LOG_INFO("Heartbeat: 00_smart_app is alive"); },
        true);

    auto api_cfg = smart_app::get_api_config();
    std::cout << "\n========================================\n"
              << "  00_smart_app is running\n"
              << "  Listen: http://" << api_cfg.host << ":" << api_cfg.port << "\n"
              << "\n"
              << "  Endpoints:\n"
              << "    GET  /healthz              Health check\n"
              << "    GET  /v1/models            List available models\n"
              << "    POST /v1/chat/completions  Chat (stream/non-stream)\n"
              << "    GET  /v1/memory            List session messages\n"
              << "    POST /v1/memory/clear      Clear session memory\n"
              << "    GET  /v1/config            View configuration\n"
              << "    POST /v1/config/reload     Hot-reload configuration\n"
              << "    POST /v1/rag/ingest        Ingest documents\n"
              << "\n"
              << "  Log file: " << log_cfg.dir << "/" << log_filename << "\n"
              << "  Press Ctrl-C to stop.\n"
              << "========================================\n\n";

    while (!g_stop.load() && app.is_running())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    LOG_INFO("Shutting down...");
    heartbeat_timer.stop();
    app.stop();
    LOG_INFO("00_smart_app stopped.");
    return 0;
}
