// chat_bubble_view.h -- WeChat-style chat bubble view for the FLTK client.
#pragma once

#include <FL/Fl_Scroll.H>

#include <string>
#include <vector>

namespace smart_app::ui
{

class ChatBubbleView : public Fl_Scroll
{
public:
    enum class Role
    {
        User,
        Assistant,
        System
    };

    ChatBubbleView(int x, int y, int w, int h, const char* label = nullptr);

    int add_message(Role role, const std::string& text);
    void append_to_message(int index, const std::string& text, bool defer_layout = false);
    void flush_layout();
    void clear_messages();

private:
    struct Message
    {
        Role role = Role::Assistant;
        std::string text;
    };

    class Canvas;

    void relayout();
    void scroll_to_bottom();

    std::vector<Message> messages_;
    Canvas* canvas_ = nullptr;
    bool layout_pending_ = false;
    bool bottom_scroll_pending_ = false;
};

} // namespace smart_app::ui
