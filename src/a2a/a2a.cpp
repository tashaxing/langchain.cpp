// src/a2a/a2a.cpp -- A2A protocol: types, client, and server.
#include "a2a/a2a.h"
#include "a2a/a2a_client.h"
#include "a2a/a2a_server.h"

#include "api/server.h"

#include <httplib.h>

#include <chrono>
#include <mutex>
#include <sstream>
#include <thread>

namespace langchain
{
namespace a2a
{

// ---------------------------------------------------------------------------
// Part serialization
// ---------------------------------------------------------------------------

json TextPart::to_json() const
{
    json j;
    j["type"] = "text";
    j["text"] = text;
    return j;
}

json FilePart::to_json() const
{
    json j;
    j["type"] = "file";
    j["name"] = name;
    j["mime_type"] = mime_type;
    j["bytes"] = bytes;
    if (!uri.empty())
    {
        j["uri"] = uri;
    }
    return j;
}

json DataPart::to_json() const
{
    json j;
    j["type"] = "data";
    j["data"] = data;
    return j;
}

std::unique_ptr<Part> Part::from_json(const json& j)
{
    if (!j.is_object())
    {
        return nullptr;
    }

    std::string type = j.value("type", std::string());
    if (type == "text")
    {
        auto p = std::make_unique<TextPart>();
        p->text = j.value("text", std::string());
        return p;
    }
    else if (type == "file")
    {
        auto p = std::make_unique<FilePart>();
        p->name     = j.value("name", std::string());
        p->mime_type = j.value("mime_type", std::string());
        p->bytes    = j.value("bytes", std::string());
        p->uri      = j.value("uri", std::string());
        return p;
    }
    else if (type == "data")
    {
        auto p = std::make_unique<DataPart>();
        if (j.contains("data"))
        {
            p->data = j["data"];
        }
        return p;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Message serialization
// ---------------------------------------------------------------------------

Message Message::text(std::string role, std::string text_content)
{
    Message m(std::move(role));
    m.parts.push_back(std::make_unique<TextPart>(std::move(text_content)));
    return m;
}

json Message::to_json() const
{
    json j;
    j["role"] = role;
    json parts_json = json::array();
    for (const auto& p : parts)
    {
        if (p)
        {
            parts_json.push_back(p->to_json());
        }
    }
    j["parts"] = std::move(parts_json);
    return j;
}

Message Message::from_json(const json& j)
{
    Message m;
    if (j.contains("role") && j["role"].is_string())
    {
        m.role = j["role"].get<std::string>();
    }
    if (j.contains("parts") && j["parts"].is_array())
    {
        for (const auto& pj : j["parts"])
        {
            auto p = Part::from_json(pj);
            if (p)
            {
                m.parts.push_back(std::move(p));
            }
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// TaskStatus
// ---------------------------------------------------------------------------

json TaskStatus::to_json() const
{
    json j;
    j["state"] = a2a::to_string(state);
    j["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
        timestamp.time_since_epoch()).count();
    return j;
}

TaskStatus TaskStatus::from_json(const json& j)
{
    TaskStatus ts;
    if (j.contains("state") && j["state"].is_string())
    {
        ts.state = task_state_from_string(j["state"].get<std::string>());
    }
    if (j.contains("timestamp"))
    {
        auto sec = j["timestamp"].get<std::time_t>();
        ts.timestamp = std::chrono::system_clock::from_time_t(sec);
    }
    else
    {
        ts.timestamp = std::chrono::system_clock::now();
    }
    return ts;
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

json Artifact::to_json() const
{
    json j;
    j["name"]        = name;
    j["description"] = description;
    j["index"]       = index;
    j["metadata"]    = metadata;
    json parts_json  = json::array();
    for (const auto& p : parts)
    {
        if (p)
        {
            parts_json.push_back(p->to_json());
        }
    }
    j["parts"] = std::move(parts_json);
    return j;
}

Artifact Artifact::from_json(const json& j)
{
    Artifact a;
    a.name        = j.value("name", std::string());
    a.description = j.value("description", std::string());
    a.index       = j.value("index", 0);
    if (j.contains("metadata") && j["metadata"].is_object())
    {
        a.metadata = j["metadata"];
    }
    if (j.contains("parts") && j["parts"].is_array())
    {
        for (const auto& pj : j["parts"])
        {
            auto p = Part::from_json(pj);
            if (p)
            {
                a.parts.push_back(std::move(p));
            }
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

json Task::to_json() const
{
    json j;
    j["id"]        = id;
    j["session_id"]= session_id;
    j["status"]    = status.to_json();

    json artifacts_json = json::array();
    for (const auto& a : artifacts)
    {
        artifacts_json.push_back(a.to_json());
    }
    j["artifacts"] = std::move(artifacts_json);

    json history_json = json::array();
    for (const auto& m : history)
    {
        history_json.push_back(m.to_json());
    }
    j["history"] = std::move(history_json);

    return j;
}

Task Task::from_json(const json& j)
{
    Task t;
    t.id         = j.value("id", std::string());
    t.session_id = j.value("session_id", std::string());
    if (j.contains("status") && j["status"].is_object())
    {
        t.status = TaskStatus::from_json(j["status"]);
    }
    if (j.contains("artifacts") && j["artifacts"].is_array())
    {
        for (const auto& aj : j["artifacts"])
        {
            t.artifacts.push_back(Artifact::from_json(aj));
        }
    }
    if (j.contains("history") && j["history"].is_array())
    {
        for (const auto& mj : j["history"])
        {
            t.history.push_back(Message::from_json(mj));
        }
    }
    return t;
}

// ---------------------------------------------------------------------------
// AgentCapability
// ---------------------------------------------------------------------------

json AgentCapability::to_json() const
{
    json j;
    j["streaming"]          = streaming;
    j["push_notifications"]  = push_notifications;
    return j;
}

AgentCapability AgentCapability::from_json(const json& j)
{
    AgentCapability c;
    c.streaming           = j.value("streaming", false);
    c.push_notifications  = j.value("push_notifications", false);
    return c;
}

// ---------------------------------------------------------------------------
// AgentAuthentication
// ---------------------------------------------------------------------------

json AgentAuthentication::to_json() const
{
    json j;
    j["schemes"] = schemes;
    return j;
}

AgentAuthentication AgentAuthentication::from_json(const json& j)
{
    AgentAuthentication a;
    if (j.contains("schemes") && j["schemes"].is_array())
    {
        for (const auto& s : j["schemes"])
        {
            if (s.is_string())
            {
                a.schemes.push_back(s.get<std::string>());
            }
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// AgentSkill
// ---------------------------------------------------------------------------

json AgentSkill::to_json() const
{
    json j;
    j["id"]          = id;
    j["name"]        = name;
    j["description"] = description;
    j["examples"]    = examples;
    j["input_modes"] = input_modes;
    j["output_modes"]= output_modes;
    return j;
}

AgentSkill AgentSkill::from_json(const json& j)
{
    AgentSkill s;
    s.id       = j.value("id", std::string());
    s.name     = j.value("name", std::string());
    s.description = j.value("description", std::string());
    if (j.contains("examples") && j["examples"].is_array())
    {
        for (const auto& e : j["examples"])
        {
            if (e.is_string())
            {
                s.examples.push_back(e.get<std::string>());
            }
        }
    }
    if (j.contains("input_modes") && j["input_modes"].is_array())
    {
        for (const auto& m : j["input_modes"])
        {
            if (m.is_string())
            {
                s.input_modes.push_back(m.get<std::string>());
            }
        }
    }
    if (j.contains("output_modes") && j["output_modes"].is_array())
    {
        for (const auto& m : j["output_modes"])
        {
            if (m.is_string())
            {
                s.output_modes.push_back(m.get<std::string>());
            }
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// AgentCard
// ---------------------------------------------------------------------------

json AgentCard::to_json() const
{
    json j;
    j["name"]           = name;
    j["description"]    = description;
    j["url"]            = url;
    j["version"]        = version;
    j["capabilities"]   = capabilities.to_json();
    j["authentication"] = authentication.to_json();
    json skills_json = json::array();
    for (const auto& s : skills)
    {
        skills_json.push_back(s.to_json());
    }
    j["skills"] = std::move(skills_json);
    return j;
}

AgentCard AgentCard::from_json(const json& j)
{
    AgentCard c;
    c.name        = j.value("name", std::string());
    c.description = j.value("description", std::string());
    c.url         = j.value("url", std::string());
    c.version     = j.value("version", std::string());
    if (j.contains("capabilities") && j["capabilities"].is_object())
    {
        c.capabilities = AgentCapability::from_json(j["capabilities"]);
    }
    if (j.contains("authentication") && j["authentication"].is_object())
    {
        c.authentication = AgentAuthentication::from_json(j["authentication"]);
    }
    if (j.contains("skills") && j["skills"].is_array())
    {
        for (const auto& sj : j["skills"])
        {
            if (sj.is_object())
            {
                c.skills.push_back(AgentSkill::from_json(sj));
            }
        }
    }
    return c;
}

AgentCard AgentCard::discover(const std::string& agent_base_url)
{
    auto pos = agent_base_url.find("://");
    std::string scheme_host = agent_base_url;
    std::string path_prefix;
    if (pos != std::string::npos)
    {
        auto path_pos = agent_base_url.find('/', pos + 3);
        if (path_pos != std::string::npos)
        {
            scheme_host = agent_base_url.substr(0, path_pos);
            path_prefix = agent_base_url.substr(path_pos);
        }
    }

    httplib::Client cli(scheme_host);
    cli.set_read_timeout(30);

    auto target = path_prefix.empty() ? std::string("/.well-known/agent.json")
                                      : path_prefix + "/.well-known/agent.json";
    auto res = cli.Get(target.c_str());
    if (!res || res->status / 100 != 2)
    {
        throw LCError("AgentCard::discover: failed to fetch agent card");
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("AgentCard::discover: invalid JSON response");
    }
    return AgentCard::from_json(j);
}

// ---------------------------------------------------------------------------
// TaskState helpers
// ---------------------------------------------------------------------------

const char* to_string(TaskState s)
{
    switch (s)
    {
        case TaskState::Pending:        return "pending";
        case TaskState::Working:        return "working";
        case TaskState::InputRequired:  return "input-required";
        case TaskState::Completed:        return "completed";
        case TaskState::Failed:         return "failed";
        case TaskState::Canceled:       return "canceled";
    }
    return "pending";
}

TaskState task_state_from_string(const std::string& s)
{
    if (s == "working")         return TaskState::Working;
    if (s == "input-required")  return TaskState::InputRequired;
    if (s == "completed")       return TaskState::Completed;
    if (s == "failed")          return TaskState::Failed;
    if (s == "canceled")        return TaskState::Canceled;
    return TaskState::Pending;
}

// ===========================================================================
// A2AClient
// ===========================================================================

namespace
{

struct ParsedUrl
{
    std::string scheme_host;
    std::string path_prefix;
};

ParsedUrl split_url(const std::string& url)
{
    ParsedUrl out;
    auto pos = url.find("://");
    if (pos == std::string::npos)
    {
        out.scheme_host = url;
        return out;
    }
    auto path_pos = url.find('/', pos + 3);
    if (path_pos == std::string::npos)
    {
        out.scheme_host = url;
        return out;
    }
    out.scheme_host = url.substr(0, path_pos);
    out.path_prefix = url.substr(path_pos);
    return out;
}

} // namespace

class A2AClient::Impl
{
public:
    std::string base_url;
    std::string agent_path;

    explicit Impl(std::string base, std::string path)
        : base_url(std::move(base)), agent_path(std::move(path))
    {
    }

    std::string full_path(const std::string& endpoint) const
    {
        if (agent_path.empty() || agent_path == "/")
        {
            return endpoint;
        }
        if (agent_path.back() == '/' && !endpoint.empty() && endpoint.front() == '/')
        {
            return agent_path + endpoint.substr(1);
        }
        return agent_path + endpoint;
    }
};

A2AClient::A2AClient(std::string base_url, std::string agent_path)
    : impl_(std::make_unique<Impl>(std::move(base_url), std::move(agent_path)))
{
}

A2AClient::~A2AClient() = default;

AgentCard A2AClient::discover() const
{
    return AgentCard::discover(impl_->base_url);
}

Task A2AClient::send_task(const Task& task) const
{
    auto parsed = split_url(impl_->base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_read_timeout(120);

    auto target = parsed.path_prefix + impl_->full_path("/tasks/send");
    auto body = task.to_json().dump();
    auto res = cli.Post(target.c_str(), body, "application/json");
    if (!res)
    {
        throw LCError("A2AClient::send_task: request failed");
    }
    if (res->status / 100 != 2)
    {
        throw LCError("A2AClient::send_task: HTTP " + std::to_string(res->status));
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("A2AClient::send_task: invalid JSON response");
    }
    return Task::from_json(j);
}

void A2AClient::send_task_stream(const Task& task,
                                 const TaskUpdateCallback& on_update) const
{
    auto parsed = split_url(impl_->base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_read_timeout(120);

    auto target = parsed.path_prefix + impl_->full_path("/tasks/sendSubscribe");
    auto body = task.to_json().dump();

    std::string buffer;
    bool aborted = false;

    auto res = cli.Post(
        target.c_str(),
        httplib::Headers{{"Content-Type", "application/json"},
                         {"Accept", "text/event-stream"}},
        body,
        "application/json",
        [&](const char* data, std::size_t len)
        {
            if (aborted)
            {
                return false;
            }
            buffer.append(data, len);
            std::size_t pos;
            while ((pos = buffer.find("\n\n")) != std::string::npos)
            {
                std::string frame = buffer.substr(0, pos);
                buffer.erase(0, pos + 2);

                std::string data_line;
                for (const auto& line : frame)
                {
                    // Simple parse: look for "data: " prefix
                }
                // Parse SSE frame: each line like "data: <json>"
                std::istringstream iss(frame);
                std::string line;
                while (std::getline(iss, line))
                {
                    auto trimmed = line;
                    // Remove trailing \r
                    if (!trimmed.empty() && trimmed.back() == '\r')
                    {
                        trimmed.pop_back();
                    }
                    if (trimmed.substr(0, 6) == "data: ")
                    {
                        std::string payload = trimmed.substr(6);
                        json j = json::parse(payload, nullptr, false);
                        if (!j.is_discarded())
                        {
                            Task update = Task::from_json(j);
                            if (!on_update(update))
                            {
                                aborted = true;
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        });

    if (!res)
    {
        throw LCError("A2AClient::send_task_stream: request failed");
    }
    if (res->status / 100 != 2)
    {
        throw LCError("A2AClient::send_task_stream: HTTP " + std::to_string(res->status));
    }
}

Task A2AClient::get_task(const std::string& task_id) const
{
    auto parsed = split_url(impl_->base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_read_timeout(30);

    auto target = parsed.path_prefix + impl_->full_path("/tasks/get?id=" + task_id);
    auto res = cli.Get(target.c_str());
    if (!res || res->status / 100 != 2)
    {
        throw LCError("A2AClient::get_task: request failed");
    }

    json j = json::parse(res->body, nullptr, false);
    if (j.is_discarded())
    {
        throw LCError("A2AClient::get_task: invalid JSON response");
    }
    return Task::from_json(j);
}

void A2AClient::cancel_task(const std::string& task_id) const
{
    auto parsed = split_url(impl_->base_url);
    httplib::Client cli(parsed.scheme_host);
    cli.set_read_timeout(30);

    json body;
    body["id"] = task_id;
    auto target = parsed.path_prefix + impl_->full_path("/tasks/cancel");
    auto res = cli.Post(target.c_str(), body.dump(), "application/json");
    if (!res || res->status / 100 != 2)
    {
        throw LCError("A2AClient::cancel_task: request failed");
    }
}

// ===========================================================================
// A2AServer
// ===========================================================================

class A2AServer::Impl
{
public:
    AgentCard card;
    TaskHandler handler;
    StreamTaskHandler stream_handler;
    std::unique_ptr<api::ApiServer> owned_server;
    std::mutex mu;

    // In-memory task store (id -> Task)
    std::unordered_map<std::string, Task> tasks;

    explicit Impl(const AgentCard& c, const TaskHandler& h)
        : card(c), handler(h)
    {
    }

    explicit Impl(const AgentCard& c, const StreamTaskHandler& h)
        : card(c), stream_handler(h)
    {
    }

    Task handle_task(const Task& incoming)
    {
        Task result = incoming;
        if (handler)
        {
            result = handler(incoming);
        }
        else if (stream_handler)
        {
            result = stream_handler(incoming,
                [&](const Task& update)
                {
                    // Streaming updates are handled by the caller
                    (void)update;
                });
        }
        std::lock_guard<std::mutex> lk(mu);
        tasks[result.id] = result;
        return result;
    }

    void handle_task_stream(const Task& incoming,
                            const std::function<void(const Task&)>& on_update,
                            std::function<void(const Task&)> on_complete)
    {
        if (stream_handler)
        {
            Task result = stream_handler(incoming, on_update);
            std::lock_guard<std::mutex> lk(mu);
            tasks[result.id] = result;
            on_complete(result);
        }
        else if (handler)
        {
            Task result = handler(incoming);
            on_update(result);
            std::lock_guard<std::mutex> lk(mu);
            tasks[result.id] = result;
            on_complete(result);
        }
        else
        {
            Task result = incoming;
            result.status.state = TaskState::Failed;
            on_complete(result);
        }
    }
};

A2AServer::A2AServer(const AgentCard& card, const TaskHandler& handler)
    : impl_(std::make_unique<Impl>(card, handler))
{
}

A2AServer::A2AServer(const AgentCard& card, const StreamTaskHandler& handler)
    : impl_(std::make_unique<Impl>(card, handler))
{
}

A2AServer::~A2AServer() = default;

void A2AServer::mount(api::ApiServer& server)
{
    // Agent card discovery endpoint
    server.add_route("GET", "/.well-known/agent.json",
        [this](const api::Request&, api::Response& res)
        {
            res.set_json(impl_->card.to_json());
        });

    // Task send endpoint
    server.add_route("POST", "/tasks/send",
        [this](const api::Request& req, api::Response& res)
        {
            json j = json::parse(req.body, nullptr, false);
            if (j.is_discarded())
            {
                res.status = 400;
                res.set_json({{"error", "invalid JSON"}});
                return;
            }
            Task incoming = Task::from_json(j);
            Task result = impl_->handle_task(incoming);
            res.set_json(result.to_json());
        });

    // Task streaming endpoint
    server.add_route("POST", "/tasks/sendSubscribe",
        [this](const api::Request& req, api::Response& res)
        {
            json j = json::parse(req.body, nullptr, false);
            if (j.is_discarded())
            {
                res.status = 400;
                res.set_json({{"error", "invalid JSON"}});
                return;
            }
            Task incoming = Task::from_json(j);

            res.status = 200;
            res.enable_streaming(
                [this, incoming](api::StreamSink& sink)
                {
                    impl_->handle_task_stream(incoming,
                        [&](const Task& update)
                        {
                            sink.write(update.to_json().dump());
                        },
                        [&](const Task& final_task)
                        {
                            (void)final_task;
                            sink.done();
                        });
                });
        });

    // Task get endpoint
    server.add_route("GET", "/tasks/get",
        [this](const api::Request& req, api::Response& res)
        {
            std::string task_id = req.query_params.count("id")
                                      ? req.query_params.at("id")
                                      : std::string();
            if (task_id.empty() && req.query_params.count("task_id"))
            {
                task_id = req.query_params.at("task_id");
            }
            if (task_id.empty())
            {
                res.status = 400;
                res.set_json({{"error", "missing task id"}});
                return;
            }

            std::lock_guard<std::mutex> lk(impl_->mu);
            auto it = impl_->tasks.find(task_id);
            if (it == impl_->tasks.end())
            {
                res.status = 404;
                res.set_json({{"error", "task not found"}});
                return;
            }
            res.set_json(it->second.to_json());
        });

    // Task cancel endpoint
    server.add_route("POST", "/tasks/cancel",
        [this](const api::Request& req, api::Response& res)
        {
            json j = json::parse(req.body, nullptr, false);
            if (j.is_discarded() || !j.contains("id"))
            {
                res.status = 400;
                res.set_json({{"error", "missing task id"}});
                return;
            }
            std::string task_id = j.value("id", std::string());
            std::lock_guard<std::mutex> lk(impl_->mu);
            auto it = impl_->tasks.find(task_id);
            if (it == impl_->tasks.end())
            {
                res.status = 404;
                res.set_json({{"error", "task not found"}});
                return;
            }
            it->second.status.state = TaskState::Canceled;
            res.set_json(it->second.to_json());
        });
}

void A2AServer::start(int port)
{
    if (impl_->owned_server && impl_->owned_server->is_running())
    {
        return;
    }
    api::ApiConfig cfg;
    cfg.port = port;
    impl_->owned_server = std::make_unique<api::ApiServer>(cfg);
    mount(*impl_->owned_server);
    impl_->owned_server->start();
}

void A2AServer::stop()
{
    if (impl_->owned_server)
    {
        impl_->owned_server->stop();
    }
}

bool A2AServer::is_running() const
{
    return impl_->owned_server && impl_->owned_server->is_running();
}

} // namespace a2a
} // namespace langchain
