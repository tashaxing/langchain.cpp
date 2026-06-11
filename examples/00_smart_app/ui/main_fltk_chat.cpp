// main_fltk_chat.cpp -- Entry point for the 00_smart_app FLTK chat client.
#include "chat_app.h"
#include "chat_config.h"

#include <FL/Fl.H>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <streambuf>

namespace
{

class NullBuffer : public std::streambuf
{
public:
    int overflow(int c) override { return c; }
};

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        std::cout.setf(std::ios::unitbuf);
        std::cerr.setf(std::ios::unitbuf);
        Fl::lock();

        auto cfg = smart_app::ui::parse_chat_config(argc, argv);

        NullBuffer null_buffer;
        std::streambuf* old_cout = nullptr;
        if (!cfg.console)
        {
            old_cout = std::cout.rdbuf(&null_buffer);
        }

        smart_app::ui::ChatApp app(cfg);
        int rc = app.run();
        if (old_cout)
        {
            std::cout.rdbuf(old_cout);
        }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "00_smart_app_chat failed: " << e.what() << "\n";
        return 1;
    }
}
