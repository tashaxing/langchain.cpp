// app_tools.cpp -- Tool registry builder.
#include "app_tools.h"
#include "app_config.h"

#include "util/logging.h"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace smart_app
{

namespace
{

// Custom tool: returns the current date and time.
class DateTimeTool : public langchain::tool::ITool
{
public:
    std::string name() const override
    {
        return "datetime";
    }

    std::string description() const override
    {
        return "Returns the current date and time in ISO 8601 format.";
    }

    langchain::json parameters_schema() const override
    {
        return {
            {"type", "object"},
            {"properties", langchain::json::object()},
            {"required", langchain::json::array()}
        };
    }

    std::string invoke(const langchain::json&) override
    {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
};

// Custom tool: echoes the input text back.
class EchoTool : public langchain::tool::ITool
{
public:
    std::string name() const override
    {
        return "echo";
    }

    std::string description() const override
    {
        return "Echoes the input text back unchanged. Useful for testing.";
    }

    langchain::json parameters_schema() const override
    {
        return {
            {"type", "object"},
            {"properties", {
                {"text", {{"type", "string"}, {"description", "Text to echo"}}}
            }},
            {"required", langchain::json::array({"text"})}
        };
    }

    std::string invoke(const langchain::json& args) override
    {
        return args.value("text", std::string());
    }
};

} // namespace

langchain::tool::ToolRegistry build_tool_registry()
{
    using namespace langchain;

    ToolConfig tcfg = get_tool_config();
    tool::ToolRegistry reg;

    if (tcfg.calculator_enabled)
    {
        reg.add(tool::make_calculator_tool());
        LOG_INFO("Tool registered: calculator");
    }

    if (tcfg.http_get_enabled)
    {
        reg.add(tool::make_http_get_tool(8192));
        LOG_INFO("Tool registered: http_get");
    }

    if (tcfg.datetime_enabled)
    {
        reg.add(std::make_shared<DateTimeTool>());
        LOG_INFO("Tool registered: datetime");
    }

    if (tcfg.echo_enabled)
    {
        reg.add(std::make_shared<EchoTool>());
        LOG_INFO("Tool registered: echo");
    }

    return reg;
}

} // namespace smart_app
