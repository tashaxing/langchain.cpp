// app_server.h -- HTTP server builder and custom route handlers.
#pragma once

#include "langchain.h"

#include <memory>
#include <string>

namespace smart_app
{

class AppServer
{
public:
    AppServer();
    ~AppServer();

    AppServer(const AppServer&) = delete;
    AppServer& operator=(const AppServer&) = delete;

    // Build the server: LLM, agent, RAG, tools, skills, memory, custom routes.
    void build();

    void start();
    void stop();
    bool is_running() const;

private:
    std::unique_ptr<langchain::api::ApiServer> server_;

    // Shared state for custom route handlers.
    std::shared_ptr<langchain::vectorstore::IVectorStore> vectorstore_;

    void setup_custom_routes();
};

} // namespace smart_app
