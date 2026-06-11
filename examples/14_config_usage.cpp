// examples/14_config_usage.cpp
// Demonstrates Config singleton: load, read, modify, save, hot-reload.

#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>

#include "util/config.h"
#include "util/singleton.h"

namespace fs = std::filesystem;
using langchain::util::Config;
using langchain::util::Singleton;

static void print_section(const Config& cfg, const std::string& name)
{
    try {
        const auto& sec = cfg.section(name);
        std::cout << "[" << name << "]\n";
        for (const auto& [k, v] : sec.snapshot()) {
            std::cout << "  " << k << " = " << v << "\n";
        }
    } catch (const std::exception&) {
        std::cout << "[" << name << "] (missing)\n";
    }
}

int main(int argc, char* argv[])
{
    std::string cfg_path = (argc > 1) ? argv[1] : "config/app_config_template.xml";

    auto& cfg = Singleton<Config>::instance();

    // ---- load ---------------------------------------------------------------
    if (!cfg.load(cfg_path)) {
        std::cerr << "Failed to load config: " << cfg_path << "\n";
        return 1;
    }
    std::cout << "Loaded config from: " << cfg.file_path() << "\n\n";

    // ---- read ---------------------------------------------------------------
    const auto& app = cfg.section("app");
    std::string name  = app.get<std::string>("name", "unknown");
    int threads       = app.get<int>("worker_threads", 1);
    std::string level = app.get("log_level", "info");

    std::cout << "App name:  " << name << "\n";
    std::cout << "Threads:   " << threads << "\n";
    std::cout << "Log level: " << level << "\n\n";

    // ---- LLM multi-source ---------------------------------------------------
    if (cfg.has_section("llm")) {
        const auto& llm = cfg.section("llm");
        std::string def = llm.get("default_source", "primary");
        std::cout << "Default LLM source: " << def << "\n";
    }

    // ---- modify at runtime --------------------------------------------------
    cfg.section("app").set("log_level", "debug");
    cfg.section("app").set("worker_threads", 8);
    std::cout << "\nModified log_level -> debug, worker_threads -> 8\n";

    // ---- save to a temporary copy -------------------------------------------
    std::string tmp = (fs::temp_directory_path() / "lc_config_usage.xml").string();
    if (cfg.save_as(tmp)) {
        std::cout << "Saved copy to: " << tmp << "\n";
    }

    // ---- hot-reload demo ----------------------------------------------------
    std::cout << "\n--- Hot-reload demo ---\n";
    std::cout << "Modify " << tmp << " in another editor, then press Enter here.\n";
    std::cin.get();

    cfg.load(tmp); // switch watch to tmp
    if (cfg.check_reload()) {
        std::cout << "Config was reloaded from disk.\n";
    } else {
        std::cout << "Config unchanged on disk.\n";
    }

    print_section(cfg, "app");

    // ---- list all sections --------------------------------------------------
    std::cout << "\nAll sections:\n";
    for (const auto& sec_name : cfg.section_names()) {
        std::cout << "  - " << sec_name << "\n";
    }

    // ---- cleanup ------------------------------------------------------------
    fs::remove(tmp);
    return 0;
}
