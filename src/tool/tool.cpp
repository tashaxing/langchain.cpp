#include "tool/tool.h"
#include "util/strings.h"

#include <httplib.h>

#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace langchain
{
namespace tool
{

// ---------------------------------------------------------------------------
// Lightweight JSON Schema validation
// ---------------------------------------------------------------------------

namespace
{

bool check_type(const json& value, const std::string& expected)
{
    if (expected == "string")
    {
        return value.is_string();
    }
    if (expected == "number")
    {
        return value.is_number();
    }
    if (expected == "integer")
    {
        return value.is_number_integer();
    }
    if (expected == "boolean")
    {
        return value.is_boolean();
    }
    if (expected == "array")
    {
        return value.is_array();
    }
    if (expected == "object")
    {
        return value.is_object();
    }
    if (expected == "null")
    {
        return value.is_null();
    }
    return true; // unknown type: permissive
}

} // namespace

ValidationResult validate_args(const json& args, const json& schema)
{
    ValidationResult result;
    if (!schema.is_object())
    {
        return result; // no schema to validate against
    }

    // Check type: args must be an object for object schemas
    if (schema.value("type", std::string()) == "object" && !args.is_object())
    {
        result.valid = false;
        result.error = "expected object, got " + std::string(args.type_name());
        return result;
    }

    // Check required fields
    if (schema.contains("required") && schema["required"].is_array())
    {
        for (const auto& req : schema["required"])
        {
            if (!req.is_string())
            {
                continue;
            }
            std::string key = req.get<std::string>();
            if (!args.contains(key))
            {
                result.valid = false;
                result.error = "missing required field: '" + key + "'";
                return result;
            }
        }
    }

    // Check property types
    if (schema.contains("properties") && schema["properties"].is_object() && args.is_object())
    {
        for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it)
        {
            std::string key = it.key();
            if (!args.contains(key))
            {
                continue; // optional field missing is OK
            }
            const json& prop_schema = it.value();
            if (prop_schema.contains("type") && prop_schema["type"].is_string())
            {
                std::string expected_type = prop_schema["type"].get<std::string>();
                if (!check_type(args[key], expected_type))
                {
                    result.valid = false;
                    result.error = "field '" + key + "' expected type '" +
                                   expected_type + "', got " + args[key].type_name();
                    return result;
                }
            }
        }
    }

    return result;
}

namespace
{

// ----- tiny recursive-descent arithmetic evaluator -----
// Grammar: expr = term (('+'|'-') term)*
//          term = factor (('*'|'/') factor)*
//          factor = number | '(' expr ')' | ('+'|'-') factor
class Calc
{
public:
    explicit Calc(const std::string& s)
        : s_(s)
    {
    }

    double parse()
    {
        double v = expr();
        skip_ws();
        if (i_ != s_.size())
        {
            throw std::runtime_error("unexpected trailing input");
        }
        return v;
    }

private:
    const std::string& s_;
    std::size_t i_ = 0;

    void skip_ws()
    {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_])))
        {
            ++i_;
        }
    }

    double expr()
    {
        double v = term();
        while (true)
        {
            skip_ws();
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-'))
            {
                char op = s_[i_++];
                double rhs = term();
                v = (op == '+') ? v + rhs : v - rhs;
            }
            else
            {
                break;
            }
        }
        return v;
    }

    double term()
    {
        double v = factor();
        while (true)
        {
            skip_ws();
            if (i_ < s_.size() && (s_[i_] == '*' || s_[i_] == '/'))
            {
                char op = s_[i_++];
                double rhs = factor();
                v = (op == '*') ? v * rhs : v / rhs;
            }
            else
            {
                break;
            }
        }
        return v;
    }

    double factor()
    {
        skip_ws();
        if (i_ >= s_.size())
        {
            throw std::runtime_error("unexpected end");
        }
        char c = s_[i_];
        if (c == '+' || c == '-')
        {
            ++i_;
            double v = factor();
            return c == '-' ? -v : v;
        }
        if (c == '(')
        {
            ++i_;
            double v = expr();
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != ')')
            {
                throw std::runtime_error("missing ')'");
            }
            ++i_;
            return v;
        }
        std::size_t start = i_;
        while (i_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.'))
        {
            ++i_;
        }
        if (start == i_)
        {
            throw std::runtime_error("expected number");
        }
        return std::stod(s_.substr(start, i_ - start));
    }
};

} // namespace

ToolPtr make_calculator_tool()
{
    json schema = {
        {"type", "object"},
        {"properties", {{"expression", {{"type", "string"},
                                        {"description", "A math expression, e.g. (1+2)*3"}}}}},
        {"required", json::array({"expression"})}
    };
    return std::make_shared<FunctionTool>(
        "calculator",
        "Evaluate a basic arithmetic expression (+ - * / and parentheses).",
        std::move(schema),
        [](const json& args) -> std::string
        {
            std::string expr = args.value("expression", std::string());
            if (expr.empty())
            {
                return "error: missing 'expression'";
            }
            try
            {
                double v = Calc(expr).parse();
                std::ostringstream oss;
                oss << v;
                return oss.str();
            }
            catch (const std::exception& e)
            {
                return std::string("error: ") + e.what();
            }
        });
}

ToolPtr make_http_get_tool(std::size_t max_bytes)
{
    json schema = {
        {"type", "object"},
        {"properties", {{"url", {{"type", "string"},
                                 {"description", "Absolute URL, e.g. https://example.com"}}}}},
        {"required", json::array({"url"})}
    };
    return std::make_shared<FunctionTool>(
        "http_get",
        "Perform an HTTP GET on the given URL and return the response body.",
        std::move(schema),
        [max_bytes](const json& args) -> std::string
        {
            std::string url = args.value("url", std::string());
            if (url.empty())
            {
                return "error: missing 'url'";
            }
            auto pos = url.find("://");
            if (pos == std::string::npos)
            {
                return "error: malformed url";
            }
            auto path_pos = url.find('/', pos + 3);
            std::string scheme_host = (path_pos == std::string::npos)
                                          ? url
                                          : url.substr(0, path_pos);
            std::string path = (path_pos == std::string::npos)
                                   ? std::string("/")
                                   : url.substr(path_pos);

            httplib::Client cli(scheme_host);
            cli.set_connection_timeout(10);
            cli.set_read_timeout(20);
            auto res = cli.Get(path.c_str());
            if (!res)
            {
                return "error: request failed";
            }
            std::string body = res->body;
            if (body.size() > max_bytes)
            {
                body.resize(max_bytes);
            }
            return "HTTP " + std::to_string(res->status) + "\n" + body;
        });
}

// ---------------- ToolRegistry::merge ----------------

void ToolRegistry::merge(const ToolRegistry& other)
{
    for (const auto& kv : other.tools_)
    {
        tools_[kv.first] = kv.second;
    }
}

} // namespace tool
} // namespace langchain
