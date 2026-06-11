// langchain/a2a/a2a.h
// Agent-to-Agent (A2A) protocol types and constants.
// Based on Google's open A2A specification (https://github.com/google/A2A).
//
// Core concepts:
//   AgentCard   -- discovery metadata describing an agent's capabilities.
//   Task        -- a unit of work sent from one agent to another.
//   Message     -- communication payload with typed parts (text, file, data).
//   Part        -- content fragment within a message.
//
// Task lifecycle states: pending -> working -> [input-required] -> completed
//                                              |-> failed
//                                              |-> canceled
#pragma once

#include "util/common.h"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace langchain
{
namespace a2a
{

// ---------------------------------------------------------------------------
// Part -- content fragment within a message.
// ---------------------------------------------------------------------------

enum class PartType
{
    Text,
    File,
    Data
};

// Base for all content parts.  Use the concrete TextPart, FilePart, DataPart.
struct Part
{
    virtual ~Part() = default;

    virtual PartType type() const = 0;
    virtual json to_json() const = 0;
    virtual Part* clone() const = 0;

    static std::unique_ptr<Part> from_json(const json& j);
};

// Plain text content.
struct TextPart : public Part
{
    std::string text;

    TextPart() = default;
    explicit TextPart(std::string t) : text(std::move(t)) {}

    PartType type() const override { return PartType::Text; }
    json to_json() const override;
    Part* clone() const override { return new TextPart(*this); }
};

// File content with metadata.
struct FilePart : public Part
{
    std::string name;
    std::string mime_type;
    std::string bytes;   // base64-encoded content
    std::string uri;     // alternative to bytes

    FilePart() = default;
    FilePart(std::string name_, std::string mime_type_,
             std::string bytes_, std::string uri_)
        : name(std::move(name_)),
          mime_type(std::move(mime_type_)),
          bytes(std::move(bytes_)),
          uri(std::move(uri_))
    {
    }

    PartType type() const override { return PartType::File; }
    json to_json() const override;
    Part* clone() const override { return new FilePart(*this); }
};

// Structured JSON data.
struct DataPart : public Part
{
    json data;

    DataPart() = default;
    explicit DataPart(json d) : data(std::move(d)) {}

    PartType type() const override { return PartType::Data; }
    json to_json() const override;
    Part* clone() const override { return new DataPart(*this); }
};

// ---------------------------------------------------------------------------
// Message -- communication payload between agents.
// ---------------------------------------------------------------------------

struct Message
{
    std::string role;   // "user" or "agent"
    std::vector<std::unique_ptr<Part>> parts;

    Message() = default;
    explicit Message(std::string r) : role(std::move(r)) {}

    // Copy: deep-copy unique_ptr elements.
    Message(const Message& other) : role(other.role)
    {
        for (const auto& p : other.parts)
        {
            if (p)
            {
                parts.push_back(std::unique_ptr<Part>(p->clone()));
            }
        }
    }
    Message& operator=(const Message& other)
    {
        if (this != &other)
        {
            role = other.role;
            parts.clear();
            for (const auto& p : other.parts)
            {
                if (p)
                {
                    parts.push_back(std::unique_ptr<Part>(p->clone()));
                }
            }
        }
        return *this;
    }

    Message(Message&&) noexcept = default;
    Message& operator=(Message&&) noexcept = default;

    // Convenience: create a text-only message.
    static Message text(std::string role, std::string text_content);

    json to_json() const;
    static Message from_json(const json& j);
};

// ---------------------------------------------------------------------------
// Task -- unit of work with state tracking.
// ---------------------------------------------------------------------------

enum class TaskState
{
    Pending,
    Working,
    InputRequired,
    Completed,
    Failed,
    Canceled
};

const char* to_string(TaskState s);
TaskState task_state_from_string(const std::string& s);

struct TaskStatus
{
    TaskState state = TaskState::Pending;
    std::chrono::system_clock::time_point timestamp;

    json to_json() const;
    static TaskStatus from_json(const json& j);
};

// Artifact produced by a task (intermediate or final output).
struct Artifact
{
    std::string name;
    std::string description;
    std::vector<std::unique_ptr<Part>> parts;
    json metadata = json::object();
    int index = 0;

    Artifact() = default;
    ~Artifact() = default;

    // Copy: deep-copy unique_ptr elements.
    Artifact(const Artifact& other)
        : name(other.name),
          description(other.description),
          metadata(other.metadata),
          index(other.index)
    {
        for (const auto& p : other.parts)
        {
            if (p)
            {
                parts.push_back(std::unique_ptr<Part>(p->clone()));
            }
        }
    }
    Artifact& operator=(const Artifact& other)
    {
        if (this != &other)
        {
            name        = other.name;
            description = other.description;
            metadata    = other.metadata;
            index       = other.index;
            parts.clear();
            for (const auto& p : other.parts)
            {
                if (p)
                {
                    parts.push_back(std::unique_ptr<Part>(p->clone()));
                }
            }
        }
        return *this;
    }

    Artifact(Artifact&&) noexcept = default;
    Artifact& operator=(Artifact&&) noexcept = default;

    json to_json() const;
    static Artifact from_json(const json& j);
};

struct Task
{
    std::string id;
    std::string session_id;
    TaskStatus status;
    std::vector<Artifact> artifacts;
    std::vector<Message> history;

    Task() = default;
    Task(const Task&) = default;
    Task& operator=(const Task&) = default;
    Task(Task&&) noexcept = default;
    Task& operator=(Task&&) noexcept = default;

    json to_json() const;
    static Task from_json(const json& j);
};

// ---------------------------------------------------------------------------
// AgentCard -- discovery metadata.
// ---------------------------------------------------------------------------

struct AgentCapability
{
    bool streaming = false;
    bool push_notifications = false;

    json to_json() const;
    static AgentCapability from_json(const json& j);
};

struct AgentAuthentication
{
    std::vector<std::string> schemes;   // e.g. "basic", "bearer", "oauth2"

    json to_json() const;
    static AgentAuthentication from_json(const json& j);
};

struct AgentSkill
{
    std::string id;
    std::string name;
    std::string description;
    std::vector<std::string> examples;
    std::vector<std::string> input_modes;   // "text", "image", "audio"
    std::vector<std::string> output_modes;  // "text", "image", "audio"

    json to_json() const;
    static AgentSkill from_json(const json& j);
};

struct AgentCard
{
    std::string name;
    std::string description;
    std::string url;
    std::string version;
    AgentCapability capabilities;
    AgentAuthentication authentication;
    std::vector<AgentSkill> skills;

    json to_json() const;
    static AgentCard from_json(const json& j);

    // Fetch an AgentCard from a remote agent's well-known endpoint.
    // `agent_base_url` -- e.g. "http://localhost:8080"
    static AgentCard discover(const std::string& agent_base_url);
};

} // namespace a2a
} // namespace langchain
