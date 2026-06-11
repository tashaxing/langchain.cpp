// langchain/tool/tool.h
// ITool + ToolRegistry + a few demo tools (calculator, http_get).
#pragma once

#include "llm/llm.h"
#include "util/common.h"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace langchain
{
namespace tool
{

// Lightweight JSON Schema validation result.
struct ValidationResult
{
    bool valid = true;
    std::string error;
};

// Validate `args` against a JSON Schema subset:
//   - checks "required" fields are present
//   - checks basic "type" (string, number, integer, boolean, array, object)
// Does NOT validate: nested schemas, pattern, min/max, enum, oneOf, etc.
ValidationResult validate_args(const json& args, const json& schema);

// A tool is something the agent can call: name, description, JSON-schema
// parameters, and a function that turns an argument JSON into a string result.
class ITool
{
public:
    virtual ~ITool() = default;

    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual json parameters_schema() const = 0;

    // Invoke with parsed arguments; return a stringified observation that will
    // be fed back to the LLM.
    virtual std::string invoke(const json& arguments) = 0;

    // Render as an OpenAI tool/function schema for ChatRequest::tools.
    llm::ToolSchema schema() const
    {
        return llm::ToolSchema{name(), description(), parameters_schema()};
    }

    // Validate arguments against parameters_schema() before invoking.
    // Default implementation delegates to validate_args().
    virtual ValidationResult validate(const json& arguments) const
    {
        return validate_args(arguments, parameters_schema());
    }
};

using ToolPtr = std::shared_ptr<ITool>;

// Quick-and-dirty tool from a lambda; mirrors langchain.tool decorator.
class FunctionTool : public ITool
{
public:
    using Fn = std::function<std::string(const json&)>;

    FunctionTool(std::string name, std::string desc, json schema, Fn fn)
        : name_(std::move(name)),
          desc_(std::move(desc)),
          schema_(std::move(schema)),
          fn_(std::move(fn))
    {
    }

    std::string name() const override
    {
        return name_;
    }
    std::string description() const override
    {
        return desc_;
    }
    json parameters_schema() const override
    {
        return schema_;
    }
    std::string invoke(const json& arguments) override
    {
        return fn_(arguments);
    }

private:
    std::string name_;
    std::string desc_;
    json schema_;
    Fn fn_;
};

class ToolRegistry
{
public:
    void add(ToolPtr t)
    {
        tools_[t->name()] = std::move(t);
    }

    // Get a tool by name. Returns nullptr if not found.
    // Callers should check the return value before use.
    ToolPtr get(const std::string& name) const
    {
        auto it = tools_.find(name);
        return it == tools_.end() ? nullptr : it->second;
    }

    // Get a tool by name, throwing LCError if not found.
    // Use this when a missing tool is a fatal error.
    ToolPtr get_or_throw(const std::string& name) const
    {
        auto it = tools_.find(name);
        if (it == tools_.end())
        {
            throw LCError("ToolRegistry: tool not found: '" + name + "'");
        }
        return it->second;
    }

    std::vector<llm::ToolSchema> schemas() const
    {
        std::vector<llm::ToolSchema> out;
        out.reserve(tools_.size());
        for (const auto& kv : tools_)
        {
            out.push_back(kv.second->schema());
        }
        return out;
    }

    std::vector<std::string> names() const
    {
        std::vector<std::string> out;
        out.reserve(tools_.size());
        for (const auto& kv : tools_)
        {
            out.push_back(kv.first);
        }
        return out;
    }

    bool empty() const
    {
        return tools_.empty();
    }

    std::size_t size() const
    {
        return tools_.size();
    }

    // Merge all tools from another registry into this one.
    void merge(const ToolRegistry& other);

private:
    std::map<std::string, ToolPtr> tools_;
};

// ---- Built-in demo tools ----

// Evaluates simple arithmetic expressions: + - * / and parentheses, doubles only.
ToolPtr make_calculator_tool();

// Performs an HTTP GET; returns body truncated to `max_bytes`.
ToolPtr make_http_get_tool(std::size_t max_bytes = 8192);

} // namespace tool
} // namespace langchain
