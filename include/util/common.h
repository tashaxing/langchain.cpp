// langchain/util/common.h
// Foundational types used across the library: Role, Message, Document, Status.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace langchain
{

using json = nlohmann::json;

enum class Role
{
    System,
    User,
    Assistant,
    Tool
};

const char* to_string(Role r);
Role role_from_string(const std::string& s);

// ---------------------------------------------------------------------------
// Multimodal content parts (OpenAI-style).
// ---------------------------------------------------------------------------
struct ContentPart
{
    std::string type; // "text" | "image_url" | "image_base64"
    std::string text; // for type="text"
    std::string url;  // for type="image_url"
    std::string base64_data; // for type="image_base64"
    std::string mime_type;   // e.g. "image/png", used with base64_data

    static ContentPart text_part(std::string t)
    {
        ContentPart p;
        p.type = "text";
        p.text = std::move(t);
        return p;
    }

    static ContentPart image_url(std::string u)
    {
        ContentPart p;
        p.type = "image_url";
        p.url = std::move(u);
        return p;
    }

    static ContentPart image_base64(std::string data, std::string mime)
    {
        ContentPart p;
        p.type = "image_base64";
        p.base64_data = std::move(data);
        p.mime_type = std::move(mime);
        return p;
    }
};

// Tool-call request emitted by an LLM (OpenAI-style function call).
struct ToolCall
{
    std::string id;        // provider-assigned correlation id
    std::string name;      // tool name to invoke
    std::string arguments; // JSON string per OpenAI spec
};

struct Message
{
    Role role{Role::User};
    std::string content;              // plain-text content (backward-compat)
    std::vector<ContentPart> content_parts; // multimodal content (preferred when non-empty)
    std::string name;                 // optional speaker name / tool name
    std::vector<ToolCall> tool_calls; // assistant->tool requests
    std::string tool_call_id;         // populated on Role::Tool replies

    Message() = default;
    Message(Role r, std::string c) : role(r), content(std::move(c)) {}

    static Message system(std::string c)
    {
        return Message(Role::System, std::move(c));
    }
    static Message user(std::string c)
    {
        return Message(Role::User, std::move(c));
    }
    static Message assistant(std::string c)
    {
        return Message(Role::Assistant, std::move(c));
    }
    static Message tool(std::string c, std::string call_id)
    {
        Message m(Role::Tool, std::move(c));
        m.tool_call_id = std::move(call_id);
        return m;
    }

    // Multimodal convenience constructors.
    static Message user_with_image(std::string text, std::string image_url)
    {
        Message m;
        m.role = Role::User;
        m.content_parts.push_back(ContentPart::text_part(std::move(text)));
        m.content_parts.push_back(ContentPart::image_url(std::move(image_url)));
        return m;
    }
    static Message user_with_image_base64(std::string text,
                                          std::string base64_data,
                                          std::string mime_type)
    {
        Message m;
        m.role = Role::User;
        m.content_parts.push_back(ContentPart::text_part(std::move(text)));
        m.content_parts.push_back(ContentPart::image_base64(std::move(base64_data), std::move(mime_type)));
        return m;
    }

    // Load a local image file and create a multimodal message.
    // `file_path` -- local filesystem path to the image.
    // `mime_type` -- e.g. "image/png", "image/jpeg". If empty, inferred from extension.
    static Message user_with_image_file(std::string text,
                                        const std::string& file_path,
                                        std::string mime_type = {});
};

// Generic retrieval document.
struct Document
{
    std::string id;
    std::string content;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<float> embedding; // optional; populated by embedders
};

// Thrown when a structured error needs to surface to the caller.
class LCError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace langchain
