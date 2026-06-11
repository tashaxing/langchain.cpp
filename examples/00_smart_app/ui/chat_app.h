// chat_app.h -- FLTK chat client window for 00_smart_app.
#pragma once

#include "chat_bubble_view.h"
#include "smart_app_client.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class Fl_Browser;
class Fl_Button;
class Fl_Check_Button;
class Fl_Choice;
class Fl_Input;
class Fl_Multiline_Input;
class Fl_Output;
class Fl_Window;

namespace smart_app::ui
{

class ChatApp
{
public:
    explicit ChatApp(ChatConfig config);
    ~ChatApp();

    int run();

private:
    void build_ui();
    void update_config_from_ui();
    void populate_initial_values();

    void on_send();
    void on_stop();
    void on_clear();
    void on_health_check();
    void on_reload_models();
    void on_select_conversation(int index);
    void on_new_conversation();

    void append_user_message(const std::string& text);
    void append_assistant_prefix();
    void append_assistant_delta(const std::string& text);
    void append_assistant_delta_deferred(const std::string& text);
    void flush_assistant_delta();
    void append_agent_event(const ChatEvent& event);
    void append_status_line(const std::string& text);
    void set_status(const std::string& text);
    void set_busy(bool busy);

    void start_worker(std::function<void()> job);
    void enqueue_ui(std::function<void()> fn);
    void drain_ui_queue();

    static void awake_cb(void* data);

private:
    ChatConfig config_;
    SmartAppClient client_;

    struct Turn
    {
        std::string role;
        std::string content;
    };
    struct Conversation
    {
        std::string title;
        std::vector<Turn> history;
    };
    std::vector<Conversation> conversations_;
    std::size_t current_conversation_ = 0;

    Fl_Window* window_ = nullptr;
    ChatBubbleView* transcript_view_ = nullptr;
    int current_assistant_bubble_ = -1;
    Fl_Multiline_Input* input_ = nullptr;
    Fl_Input* host_input_ = nullptr;
    Fl_Input* port_input_ = nullptr;
    Fl_Input* session_input_ = nullptr;
    Fl_Input* temperature_input_ = nullptr;
    Fl_Input* top_p_input_ = nullptr;
    Fl_Input* top_k_input_ = nullptr;
    Fl_Input* max_tokens_input_ = nullptr;
    Fl_Browser* conv_list_ = nullptr;
    Fl_Choice* model_choice_ = nullptr;
    Fl_Check_Button* stream_check_ = nullptr;
    Fl_Button* send_button_ = nullptr;
    Fl_Button* stop_button_ = nullptr;
    Fl_Button* clear_button_ = nullptr;
    Fl_Button* health_button_ = nullptr;
    Fl_Button* reload_models_button_ = nullptr;
    Fl_Button* new_conv_button_ = nullptr;
    Fl_Output* status_output_ = nullptr;

    std::thread worker_;
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> worker_running_{false};

    std::string pending_assistant_delta_;
    bool assistant_flush_scheduled_ = false;

    std::mutex queue_mutex_;
    std::queue<std::function<void()>> ui_queue_;
};

} // namespace smart_app::ui
