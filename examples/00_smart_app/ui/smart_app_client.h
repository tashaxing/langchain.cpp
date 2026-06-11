// smart_app_client.h -- HTTP client wrapper for 00_smart_app chat UI.
#pragma once

#include "chat_config.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace smart_app::ui
{

struct ModelInfo
{
    std::string id;
};

struct ChatMessage
{
    std::string role;
    std::string content;
};

struct ChatEvent
{
    enum class Type
    {
        Delta,
        AgentEvent,
        Done,
        Error
    };

    Type type = Type::Delta;
    std::string text;
    std::string agent_event_type;
    std::string tool_name;
    std::string tool_input;
    std::string raw_json;
};

class SmartAppClient
{
public:
    explicit SmartAppClient(ChatConfig config = {});

    ChatConfig config() const;
    void set_config(ChatConfig config);

    bool health(std::string* error = nullptr) const;
    std::vector<ModelInfo> list_models(std::string* error = nullptr) const;

    std::string chat_once(const std::vector<ChatMessage>& messages,
                          std::string* error = nullptr) const;

    bool chat_stream(const std::vector<ChatMessage>& messages,
                     const std::function<bool(const ChatEvent&)>& on_event,
                     std::string* error = nullptr) const;

    bool clear_memory(std::string* error = nullptr) const;

private:
    ChatConfig config_;
};

} // namespace smart_app::ui
