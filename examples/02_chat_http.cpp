// examples/02_chat_http.cpp — single-shot and multi-turn chat against an
// OpenAI-compatible endpoint, demonstrating short-term (in-memory) BufferMemory.
// Run with env: LC_BASE_URL, LC_API_KEY, LC_MODEL.
#include "langchain.h"

#include <cstdlib>
#include <iostream>

int main()
{
    using namespace langchain;

    llm::OpenAILLMConfig cfg;
    if (const char* e = std::getenv("LC_BASE_URL"))
    {
        cfg.base_url = e;
    }
    if (const char* e = std::getenv("LC_API_KEY"))
    {
        cfg.api_key = e;
    }
    if (const char* e = std::getenv("LC_MODEL"))
    {
        cfg.model = e;
    }

    llm::OpenAILLM client(cfg);

    // ---- Short-term memory: keeps every exchange in memory ----
    memory::BufferMemory mem;

    try
    {
        // Turn 1
        {
            llm::ChatRequest req;
            req.messages.push_back(Message::system("You are a concise C++ assistant."));
            for (const auto& m : mem.messages())
            {
                req.messages.push_back(m);
            }
            req.messages.push_back(
                Message::user("In one sentence, what does std::span do?"));
            req.temperature = 0.2f;

            auto resp = client.invoke(req);
            std::cout << "[Turn 1] " << resp.model << ": " << resp.message.content << "\n";
            mem.add_exchange("In one sentence, what does std::span do?",
                             resp.message.content);
        }

        // Turn 2 — the LLM sees the previous exchange because mem is replayed.
        {
            llm::ChatRequest req;
            req.messages.push_back(Message::system("You are a concise C++ assistant."));
            for (const auto& m : mem.messages())
            {
                req.messages.push_back(m);
            }
            req.messages.push_back(
                Message::user("Give me a short code example using it."));
            req.temperature = 0.2f;

            auto resp = client.invoke(req);
            std::cout << "[Turn 2] " << resp.model << ": " << resp.message.content << "\n";
            mem.add_exchange("Give me a short code example using it.",
                             resp.message.content);
        }

        std::cout << "Memory holds " << mem.messages().size() << " messages.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "chat failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
