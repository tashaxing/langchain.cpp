// chat_app.cpp -- FLTK chat client window for 00_smart_app.
#include "chat_app.h"

#include <FL/Fl.H>
#include <FL/Fl_Browser.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Multiline_Input.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Window.H>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>

namespace smart_app::ui
{

namespace
{

std::string widget_value(Fl_Input* w)
{
    const char* v = w ? w->value() : nullptr;
    return v ? std::string(v) : std::string();
}

std::string widget_value(Fl_Multiline_Input* w)
{
    const char* v = w ? w->value() : nullptr;
    return v ? std::string(v) : std::string();
}

std::string selected_choice_text(Fl_Choice* choice)
{
    if (!choice || choice->value() < 0)
    {
        return {};
    }
    const char* text = choice->text(choice->value());
    return text ? std::string(text) : std::string();
}

} // namespace

ChatApp::ChatApp(ChatConfig config)
    : config_(std::move(config)), client_(config_)
{
    build_ui();
    populate_initial_values();
}

ChatApp::~ChatApp()
{
    cancel_requested_.store(true);
    if (worker_.joinable())
    {
        worker_.join();
    }
    delete window_;
}

int ChatApp::run()
{
    window_->show();
    Fl::awake(awake_cb, this);
    Fl::add_timeout(0.2, [](void* data) {
        static_cast<ChatApp*>(data)->on_reload_models();
    }, this);
    return Fl::run();
}

void ChatApp::build_ui()
{
    window_ = new Fl_Window(1180, 720, "00_smart_app Chat");

    // ---- Connection bar ----
    host_input_ = new Fl_Input(70, 12, 200, 26, "Host:");
    port_input_ = new Fl_Input(310, 12, 60, 26, "Port:");
    health_button_ = new Fl_Button(410, 12, 85, 26, "Health");
    reload_models_button_ = new Fl_Button(500, 12, 90, 26, "Models");

    // ---- Model / session bar ----
    session_input_ = new Fl_Input(70, 48, 190, 26, "Session:");
    model_choice_ = new Fl_Choice(360, 48, 300, 26, "Model:");
    stream_check_ = new Fl_Check_Button(685, 50, 90, 22, "Stream");

    // ---- Generation parameter bar ----
    temperature_input_ = new Fl_Input(110, 84, 70, 26, "Temperature:");
    top_p_input_ = new Fl_Input(260, 84, 70, 26, "Top P:");
    top_k_input_ = new Fl_Input(405, 84, 70, 26, "Top K:");
    max_tokens_input_ = new Fl_Input(595, 84, 80, 26, "Max Tokens:");

    // ---- Conversation list (left sidebar) ----
    conv_list_ = new Fl_Browser(12, 122, 200, 464);
    conv_list_->callback([](Fl_Widget*, void* data) {
        auto* app = static_cast<ChatApp*>(data);
        if (app->conv_list_->value() > 0)
        {
            app->on_select_conversation(app->conv_list_->value() - 1);
        }
    }, this);

    new_conv_button_ = new Fl_Button(12, 594, 200, 30, "New Conversation");
    new_conv_button_->label("New Conversation");
    new_conv_button_->callback([](Fl_Widget*, void* data) {
        static_cast<ChatApp*>(data)->on_new_conversation();
    }, this);

    // ---- Transcript (main area) ----
    transcript_view_ = new ChatBubbleView(224, 122, 944, 464);

    // ---- Input area ----
    input_ = new Fl_Multiline_Input(224, 598, 680, 76);
    send_button_ = new Fl_Button(920, 598, 75, 32, "Send");
    stop_button_ = new Fl_Button(1005, 598, 75, 32, "Stop");
    clear_button_ = new Fl_Button(920, 642, 160, 32, "Clear Session");
    status_output_ = new Fl_Output(280, 684, 888, 24, "Status:");

    // ---- Button callbacks ----
    health_button_->callback([](Fl_Widget*, void* data) {
        static_cast<ChatApp*>(data)->on_health_check();
    }, this);
    reload_models_button_->callback([](Fl_Widget*, void* data) {
        static_cast<ChatApp*>(data)->on_reload_models();
    }, this);
    send_button_->callback([](Fl_Widget*, void* data) {
        static_cast<ChatApp*>(data)->on_send();
    }, this);
    stop_button_->callback([](Fl_Widget*, void* data) {
        static_cast<ChatApp*>(data)->on_stop();
    }, this);
    clear_button_->callback([](Fl_Widget*, void* data) {
        static_cast<ChatApp*>(data)->on_clear();
    }, this);

    stop_button_->deactivate();
    window_->end();
    window_->resizable(transcript_view_);
}

void ChatApp::populate_initial_values()
{
    host_input_->value(config_.host.c_str());
    std::ostringstream port;
    port << config_.port;
    port_input_->value(port.str().c_str());
    session_input_->value(config_.session_id.c_str());
    stream_check_->value(config_.stream ? 1 : 0);

    std::ostringstream temp;
    temp << config_.temperature;
    temperature_input_->value(temp.str().c_str());

    std::ostringstream top_p;
    top_p << config_.top_p;
    top_p_input_->value(top_p.str().c_str());

    std::ostringstream top_k;
    top_k << config_.top_k;
    top_k_input_->value(top_k.str().c_str());

    std::ostringstream max_tokens;
    max_tokens << config_.max_tokens;
    max_tokens_input_->value(max_tokens.str().c_str());

    model_choice_->clear();
    int selected = 0;
    for (std::size_t i = 0; i < config_.models.size(); ++i)
    {
        model_choice_->add(config_.models[i].c_str());
        if (!config_.model.empty() && config_.models[i] == config_.model)
        {
            selected = static_cast<int>(i);
        }
    }
    if (!config_.models.empty())
    {
        model_choice_->value(selected);
        config_.model = config_.models[static_cast<std::size_t>(selected)];
    }
    client_.set_config(config_);
    on_new_conversation();
    set_status("Ready");
}

void ChatApp::update_config_from_ui()
{
    config_.host = widget_value(host_input_);
    try
    {
        config_.port = std::stoi(widget_value(port_input_));
    }
    catch (...)
    {
        config_.port = 8080;
        port_input_->value("8080");
    }
    config_.session_id = widget_value(session_input_);
    config_.stream = stream_check_->value() != 0;

    std::string model = selected_choice_text(model_choice_);
    if (!model.empty())
    {
        config_.model = model;
    }

    try
    {
        config_.temperature = std::stof(widget_value(temperature_input_));
    }
    catch (...)
    {
        config_.temperature = 0.7f;
        temperature_input_->value("0.7");
    }

    try
    {
        config_.top_p = std::stof(widget_value(top_p_input_));
    }
    catch (...)
    {
        config_.top_p = 0.95f;
        top_p_input_->value("0.95");
    }

    try
    {
        config_.top_k = std::stoi(widget_value(top_k_input_));
    }
    catch (...)
    {
        config_.top_k = 50;
        top_k_input_->value("50");
    }

    try
    {
        config_.max_tokens = std::stoi(widget_value(max_tokens_input_));
    }
    catch (...)
    {
        config_.max_tokens = 1024;
        max_tokens_input_->value("1024");
    }

    client_.set_config(config_);
}

void ChatApp::on_send()
{
    if (worker_running_.load())
    {
        return;
    }

    update_config_from_ui();
    std::string text = widget_value(input_);
    if (text.empty())
    {
        set_status("Enter a message first");
        std::cout << "[chat-ui] Send ignored: empty input" << std::endl;
        return;
    }
    if (config_.model.empty() && !config_.models.empty())
    {
        config_.model = config_.models.front();
        model_choice_->value(0);
        client_.set_config(config_);
    }
    if (config_.model.empty())
    {
        set_status("No model configured in chat_client_config.xml");
        append_status_line("请先在客户端配置文件 chat_client_config.xml 的 <models> 中配置可用模型。");
        return;
    }

    std::cout << "[chat-ui] Send request host=" << config_.host
              << " port=" << config_.port
              << " model=" << config_.model
              << " session=" << config_.session_id
              << " stream=" << (config_.stream ? "true" : "false")
              << " temperature=" << config_.temperature
              << " top_p=" << config_.top_p
              << " top_k=" << config_.top_k
              << " max_tokens=" << config_.max_tokens
              << " text_len=" << text.size() << std::endl;

    // Ensure at least one conversation exists.
    if (conversations_.empty())
    {
        on_new_conversation();
    }

    input_->value("");
    append_user_message(text);

    auto& conv = conversations_[current_conversation_];
    conv.history.push_back({"user", text});
    set_busy(true);
    cancel_requested_.store(false);

    auto messages = std::vector<ChatMessage>{};
    for (const auto& turn : conv.history)
    {
        messages.push_back({turn.role, turn.content});
    }
    auto stream = config_.stream;
    start_worker([this, messages, stream]() mutable {
        std::string assistant;
        std::string error;

        enqueue_ui([this] { append_assistant_prefix(); });

        if (stream)
        {
            std::cout << "[chat-ui] POST /v1/chat/completions stream=true" << std::endl;
            bool ok = client_.chat_stream(messages,
                [&](const ChatEvent& ev) -> bool
            {
                if (cancel_requested_.load())
                {
                    return false;
                }
                if (ev.type == ChatEvent::Type::Delta && !ev.text.empty())
                {
                    assistant += ev.text;
                    std::string delta = ev.text;
                    enqueue_ui([this, delta] { append_assistant_delta_deferred(delta); });
                }
                else if (ev.type == ChatEvent::Type::AgentEvent)
                {
                    ChatEvent copy = ev;
                    if (copy.agent_event_type == "answer" && !copy.text.empty())
                    {
                        assistant += copy.text;
                    }
                    enqueue_ui([this, copy] { append_agent_event(copy); });
                }
                else if (ev.type == ChatEvent::Type::Error)
                {
                    std::string msg = ev.text;
                    enqueue_ui([this, msg] { append_status_line("流式数据解析失败，请稍后重试。详情：" + msg); });
                }
                return true;
            }, &error);

            std::cout << "[chat-ui] stream result ok=" << (ok ? "true" : "false")
                      << " assistant_len=" << assistant.size()
                      << " error=" << error << std::endl;
            enqueue_ui([this, assistant, ok, error] {
                flush_assistant_delta();
                if (!assistant.empty())
                {
                    conversations_[current_conversation_].history.push_back({"assistant", assistant});
                }
                else if (ok)
                {
                    append_assistant_delta("服务端已结束应答，但没有返回文本内容。请换个问题再试一次。");
                }
                else
                {
                    append_assistant_delta("抱歉，这次连接后端智能体服务失败了。请确认服务已启动、IP/端口正确后重试。");
                    if (!error.empty())
                    {
                        append_status_line("错误详情：" + error);
                    }
                }
                current_assistant_bubble_ = -1;
                set_status(ok ? "Done" : "Request failed");
                set_busy(false);
            });
        }
        else
        {
            std::cout << "[chat-ui] POST /v1/chat/completions stream=false" << std::endl;
            std::string answer = client_.chat_once(messages, &error);
            std::cout << "[chat-ui] non-stream answer_len=" << answer.size()
                      << " error=" << error << std::endl;
            enqueue_ui([this, answer, error] {
                if (!answer.empty())
                {
                    append_assistant_delta(answer);
                    conversations_[current_conversation_].history.push_back({"assistant", answer});
                    set_status("Done");
                }
                else
                {
                    append_assistant_delta("抱歉，这次连接后端智能体服务失败了。请确认服务已启动、IP/端口正确后重试。");
                    if (!error.empty())
                    {
                        append_status_line("错误详情：" + error);
                    }
                    set_status("Request failed");
                }
                current_assistant_bubble_ = -1;
                set_busy(false);
            });
        }
    });
}

void ChatApp::on_stop()
{
    cancel_requested_.store(true);
    set_status("Stopping...");
}

void ChatApp::on_clear()
{
    if (worker_running_.load())
    {
        set_status("Stop current request before clearing");
        return;
    }

    update_config_from_ui();
    transcript_view_->clear_messages();
    conversations_[current_conversation_].history.clear();
    set_status("Clearing session memory...");

    start_worker([this] {
        std::string error;
        bool ok = client_.clear_memory(&error);
        enqueue_ui([this, ok, error] {
            set_status(ok ? "Session cleared" : ("Clear failed: " + error));
        });
    });
}

void ChatApp::on_health_check()
{
    if (worker_running_.load())
    {
        return;
    }
    update_config_from_ui();
    set_status("Checking health...");
    std::cout << "[chat-ui] GET /healthz host=" << config_.host
              << " port=" << config_.port << std::endl;

    start_worker([this] {
        std::string error;
        bool ok = client_.health(&error);
        std::cout << "[chat-ui] health result ok=" << (ok ? "true" : "false")
                  << " error=" << error << std::endl;
        enqueue_ui([this, ok, error] {
            set_status(ok ? "Connected" : ("Health failed: " + error));
        });
    });
}

void ChatApp::on_reload_models()
{
    update_config_from_ui();
    model_choice_->clear();

    if (config_.models.empty())
    {
        set_status("No models configured in chat_client_config.xml");
        std::cout << "[chat-ui] no client-side models configured" << std::endl;
        return;
    }

    int selected = 0;
    for (std::size_t i = 0; i < config_.models.size(); ++i)
    {
        model_choice_->add(config_.models[i].c_str());
        if (config_.models[i] == config_.model)
        {
            selected = static_cast<int>(i);
        }
    }
    model_choice_->value(selected);
    config_.model = config_.models[static_cast<std::size_t>(selected)];
    client_.set_config(config_);
    set_status("Models loaded from client config");
    std::cout << "[chat-ui] loaded " << config_.models.size()
              << " models from chat_client_config.xml" << std::endl;
}

void ChatApp::on_new_conversation()
{
    Conversation conv;
    conv.title = "Chat " + std::to_string(conversations_.size() + 1);
    conversations_.push_back(std::move(conv));
    current_conversation_ = conversations_.size() - 1;

    conv_list_->add(conversations_[current_conversation_].title.c_str());
    conv_list_->value(static_cast<int>(current_conversation_) + 1);

    transcript_view_->clear_messages();
    set_status("New conversation started");
}

void ChatApp::on_select_conversation(int index)
{
    if (index < 0 || static_cast<std::size_t>(index) >= conversations_.size())
    {
        return;
    }
    current_conversation_ = static_cast<std::size_t>(index);

    transcript_view_->clear_messages();
    const auto& conv = conversations_[current_conversation_];
    for (const auto& turn : conv.history)
    {
        if (turn.role == "user")
        {
            append_user_message(turn.content);
        }
        else if (turn.role == "assistant")
        {
            append_assistant_prefix();
            append_assistant_delta(turn.content);
            current_assistant_bubble_ = -1;
        }
    }
}

void ChatApp::append_user_message(const std::string& text)
{
    transcript_view_->add_message(ChatBubbleView::Role::User, text);
}

void ChatApp::append_assistant_prefix()
{
    current_assistant_bubble_ = transcript_view_->add_message(ChatBubbleView::Role::Assistant, "");
}

void ChatApp::append_assistant_delta(const std::string& text)
{
    if (current_assistant_bubble_ < 0)
    {
        append_assistant_prefix();
    }
    transcript_view_->append_to_message(current_assistant_bubble_, text);
}

void ChatApp::append_assistant_delta_deferred(const std::string& text)
{
    if (current_assistant_bubble_ < 0)
    {
        append_assistant_prefix();
    }
    pending_assistant_delta_ += text;
    if (assistant_flush_scheduled_)
    {
        return;
    }
    assistant_flush_scheduled_ = true;
    Fl::add_timeout(0.05, [](void* data) {
        static_cast<ChatApp*>(data)->flush_assistant_delta();
    }, this);
}

void ChatApp::flush_assistant_delta()
{
    assistant_flush_scheduled_ = false;
    if (pending_assistant_delta_.empty())
    {
        return;
    }
    std::string delta;
    delta.swap(pending_assistant_delta_);
    if (current_assistant_bubble_ < 0)
    {
        append_assistant_prefix();
    }
    transcript_view_->append_to_message(current_assistant_bubble_, delta, true);
    transcript_view_->flush_layout();
}

void ChatApp::append_agent_event(const ChatEvent& event)
{
    if (event.agent_event_type == "done")
    {
        return;
    }
    if (event.agent_event_type == "thought" || event.agent_event_type == "answer")
    {
        if (!event.text.empty())
        {
            append_assistant_delta(event.text);
        }
        return;
    }

    std::string line = "[" + event.agent_event_type + "]";
    if (!event.tool_name.empty())
    {
        line += " " + event.tool_name;
    }
    if (!event.tool_input.empty())
    {
        line += " " + event.tool_input;
    }
    if (!event.text.empty())
    {
        line += " " + event.text;
    }
    transcript_view_->add_message(ChatBubbleView::Role::System, line);
}

void ChatApp::append_status_line(const std::string& text)
{
    transcript_view_->add_message(ChatBubbleView::Role::System, text);
}

void ChatApp::set_status(const std::string& text)
{
    if (status_output_)
    {
        status_output_->value(text.c_str());
    }
}

void ChatApp::set_busy(bool busy)
{
    if (busy)
    {
        send_button_->deactivate();
        health_button_->deactivate();
        reload_models_button_->deactivate();
        clear_button_->deactivate();
        stop_button_->activate();
    }
    else
    {
        send_button_->activate();
        health_button_->activate();
        reload_models_button_->activate();
        clear_button_->activate();
        stop_button_->deactivate();
        worker_running_.store(false);
        if (worker_.joinable())
        {
            worker_.join();
        }
    }
}

void ChatApp::start_worker(std::function<void()> job)
{
    if (worker_.joinable())
    {
        worker_.join();
    }
    worker_running_.store(true);
    worker_ = std::thread([this, job = std::move(job)]() mutable {
        job();
        enqueue_ui([this] {
            if (worker_running_.load())
            {
                set_busy(false);
            }
        });
    });
}

void ChatApp::enqueue_ui(std::function<void()> fn)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        ui_queue_.push(std::move(fn));
    }
    Fl::awake(awake_cb, this);
}

void ChatApp::drain_ui_queue()
{
    std::queue<std::function<void()>> local;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::swap(local, ui_queue_);
    }

    while (!local.empty())
    {
        local.front()();
        local.pop();
    }
}

void ChatApp::awake_cb(void* data)
{
    static_cast<ChatApp*>(data)->drain_ui_queue();
}

} // namespace smart_app::ui
