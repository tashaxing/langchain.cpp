// langchain/prompt/prompt_template.h
// PromptTemplate: mustache-style {var} substitution. Mirrors langchain.PromptTemplate.
#pragma once

#include "util/common.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace langchain
{
namespace prompt
{

class PromptTemplate
{
public:
    // Build from a template string like "Hello {name}, today is {date}".
    explicit PromptTemplate(std::string tmpl);

    // Names of every {placeholder} discovered in the template.
    const std::vector<std::string>& input_variables() const
    {
        return input_variables_;
    }

    // Render with the provided values. Missing keys raise LCError unless
    // allow_missing=true (then the placeholder is kept verbatim).
    std::string format(const std::unordered_map<std::string, std::string>& vars,
                       bool allow_missing = false) const;

    const std::string& raw() const
    {
        return template_;
    }

private:
    std::string template_;
    std::vector<std::string> input_variables_;
};

// Build a chat-style prompt from per-role templates. Resembles
// langchain.ChatPromptTemplate but kept intentionally minimal.
class ChatPromptTemplate
{
public:
    struct Slot
    {
        Role role;
        PromptTemplate tmpl;
    };

    explicit ChatPromptTemplate(std::vector<Slot> slots)
        : slots_(std::move(slots))
    {
    }

    std::vector<Message> format_messages(
        const std::unordered_map<std::string, std::string>& vars) const;

private:
    std::vector<Slot> slots_;
};

} // namespace prompt
} // namespace langchain
