// chat_config.cpp -- Local configuration for the FLTK chat client.
#include "chat_config.h"

#include <pugixml.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace smart_app::ui
{

namespace
{

fs::path executable_path(char* argv0)
{
#ifdef _WIN32
    char buffer[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (len > 0 && len < MAX_PATH)
    {
        return fs::path(buffer);
    }
#endif
    return fs::absolute(argv0 ? fs::path(argv0) : fs::path());
}

fs::path default_config_path(char* argv0)
{
    auto exe = executable_path(argv0);
    auto bin_dir = exe.parent_path();
    auto app_dir = bin_dir.parent_path();
    return app_dir / "config" / "chat_client_config.xml";
}

void load_config_file(ChatConfig& cfg, const fs::path& path)
{
    if (path.empty() || !fs::exists(path))
    {
        return;
    }

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.string().c_str());
    if (!result)
    {
        std::cerr << "Warning: failed to load chat client config " << path.string()
                  << ": " << result.description() << "\n";
        return;
    }

    auto root = doc.child("chat_client");

    auto server = root.child("server");
    if (server)
    {
        cfg.host = server.child("host").text().as_string(cfg.host.c_str());
        cfg.port = server.child("port").text().as_int(cfg.port);
        cfg.connect_timeout_sec = server.child("connect_timeout_sec").text().as_int(cfg.connect_timeout_sec);
        cfg.read_timeout_sec = server.child("read_timeout_sec").text().as_int(cfg.read_timeout_sec);
    }

    auto models = root.child("models");
    if (models)
    {
        cfg.models.clear();
        cfg.model = models.attribute("default").as_string(cfg.model.c_str());
        for (auto node : models.children("model"))
        {
            std::string id = node.text().as_string();
            if (!id.empty())
            {
                cfg.models.push_back(id);
            }
        }
        if (cfg.model.empty() && !cfg.models.empty())
        {
            cfg.model = cfg.models.front();
        }
    }

    auto ui = root.child("ui");
    if (ui)
    {
        cfg.console = ui.child("console").text().as_bool(cfg.console);
    }

    auto chat = root.child("chat");
    if (chat)
    {
        cfg.session_id = chat.child("session_id").text().as_string(cfg.session_id.c_str());
        cfg.stream = chat.child("stream").text().as_bool(cfg.stream);
        cfg.temperature = chat.child("temperature").text().as_float(cfg.temperature);
        cfg.top_p = chat.child("top_p").text().as_float(cfg.top_p);
        cfg.top_k = chat.child("top_k").text().as_int(cfg.top_k);
        cfg.max_tokens = chat.child("max_tokens").text().as_int(cfg.max_tokens);
    }
}

} // namespace

std::string ChatConfig::base_url() const
{
    std::ostringstream oss;
    oss << "http://" << host << ":" << port;
    return oss.str();
}

ChatConfig parse_chat_config(int argc, char* argv[])
{
    ChatConfig cfg;
    fs::path config_path = default_config_path(argc > 0 ? argv[0] : nullptr);

    // First pass: allow --config to choose the local connection config file.
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "--config")
        {
            config_path = argv[i + 1];
            break;
        }
    }

    load_config_file(cfg, config_path);

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
    {
        args.emplace_back(argv[i]);
    }

    for (std::size_t i = 0; i < args.size(); ++i)
    {
        const auto& a = args[i];
        if (a == "--config" && i + 1 < args.size())
        {
            ++i;
        }
        else if (a == "--host" && i + 1 < args.size())
        {
            cfg.host = args[++i];
        }
        else if (a == "--port" && i + 1 < args.size())
        {
            cfg.port = std::stoi(args[++i]);
        }
        else if (a == "--model" && i + 1 < args.size())
        {
            cfg.model = args[++i];
        }
        else if (a == "--session" && i + 1 < args.size())
        {
            cfg.session_id = args[++i];
        }
        else if (a == "--console")
        {
            cfg.console = true;
        }
        else if (a == "--no-console")
        {
            cfg.console = false;
        }
        else if (a == "--no-stream")
        {
            cfg.stream = false;
        }
        else if (a == "--temperature" && i + 1 < args.size())
        {
            cfg.temperature = std::stof(args[++i]);
        }
        else if (a == "--top-p" && i + 1 < args.size())
        {
            cfg.top_p = std::stof(args[++i]);
        }
        else if (a == "--top-k" && i + 1 < args.size())
        {
            cfg.top_k = std::stoi(args[++i]);
        }
        else if (a == "--max-tokens" && i + 1 < args.size())
        {
            cfg.max_tokens = std::stoi(args[++i]);
        }
        else if (a == "--connect-timeout" && i + 1 < args.size())
        {
            cfg.connect_timeout_sec = std::stoi(args[++i]);
        }
        else if (a == "--read-timeout" && i + 1 < args.size())
        {
            cfg.read_timeout_sec = std::stoi(args[++i]);
        }
        else if (a == "--help" || a == "-h")
        {
            print_chat_usage(argv[0]);
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << a << "\n";
            print_chat_usage(argv[0]);
            std::exit(1);
        }
    }

    return cfg;
}

void print_chat_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --config <path>            Local client config XML\n"
              << "  --host <ip-or-host>        Backend agent server host\n"
              << "  --port <port>              Backend agent server port\n"
              << "  --model <id>               Initial model id\n"
              << "  --session <id>             Session id for X-Session-Id (default: desktop)\n"
              << "  --console                  Enable console logs\n"
              << "  --no-console               Disable console logs\n"
              << "  --no-stream                Use non-streaming chat\n"
              << "  --temperature <float>      Sampling temperature (default: 0.7)\n"
              << "  --top-p <float>            Nucleus sampling top_p (default: 0.95)\n"
              << "  --top-k <int>              Top-k sampling value (default: 50)\n"
              << "  --max-tokens <int>         Max completion tokens (default: 1024)\n"
              << "  --connect-timeout <sec>    Connection timeout (default: 10)\n"
              << "  --read-timeout <sec>       Read timeout (default: 300)\n"
              << "  --help                     Show this help\n\n"
              << "Default config: <app_base>/config/chat_client_config.xml\n"
              << "Config sections:\n"
              << "  <server> host, port, connect_timeout_sec, read_timeout_sec\n"
              << "  <models default=\"...\"><model>...</model></models>\n"
              << "  <ui> console\n"
              << "  <chat> session_id, stream, temperature, top_p, top_k, max_tokens\n";
}

} // namespace smart_app::ui
