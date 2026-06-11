// chat_config.h -- Local configuration for the FLTK chat client.
#pragma once

#include <string>
#include <vector>

namespace smart_app::ui
{

struct ChatConfig
{
    std::string host = "127.0.0.1";
    int         port = 8080;
    std::vector<std::string> models;
    std::string model;
    std::string session_id = "desktop";
    bool        console = true;
    bool        stream = true;
    float       temperature = 0.7f;
    float       top_p = 0.95f;
    int         top_k = 50;
    int         max_tokens = 1024;
    int         connect_timeout_sec = 10;
    int         read_timeout_sec = 300;

    std::string base_url() const;
};

// Parse local XML config and optional CLI overrides.
// Default config path: <app_base>/config/chat_client_config.xml.
ChatConfig parse_chat_config(int argc, char* argv[]);

// Print usage to stdout.
void print_chat_usage(const char* prog);

} // namespace smart_app::ui
