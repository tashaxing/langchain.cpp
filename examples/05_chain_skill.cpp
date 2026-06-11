// examples/05_chain_skill.cpp — compose two skills: summarize then translate.
// Demonstrates how a ChainSkill can use a shared JsonFileMemory so that each
// step (and future runs) retains context.
// Run with env: LC_BASE_URL, LC_API_KEY, LC_MODEL.
#include "langchain.h"

#include <cstdlib>
#include <filesystem>
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
    auto llm = std::make_shared<llm::OpenAILLM>(cfg);

    // ---- Persistent memory shared across the chain ----
    // LongTermMemory with JSON file backend stores one JSON object per line,
    // making it easy to inspect or version-control conversation history.
    std::filesystem::create_directories("build");
    auto mem = std::make_shared<memory::LongTermMemory>(
        memory::LongTermMemory::json_file("build/chain_skill_memory.json", "demo-chain"));

    auto summarize = std::make_shared<skill::PromptSkill>(
        "summarize", "Summarize input text in one sentence.",
        llm,
        prompt::PromptTemplate("Summarize this in one sentence:\n{text}"));

    auto translate = std::make_shared<skill::PromptSkill>(
        "translate_zh", "Translate English text to Chinese.",
        llm,
        prompt::PromptTemplate("Translate to Chinese, output only the translation:\n{summary}"));

    skill::ChainSkill chain(
        "summarize_then_translate",
        "Summarize, then translate the summary to Chinese.",
        {{summarize, "summary"}, {translate, "translated"}});

    skill::SkillContext ctx;
    ctx.vars["text"] =
        "C++20 introduced modules, coroutines, concepts, and ranges, "
        "fundamentally modernizing how we structure C++ codebases.";

    try
    {
        auto trace = chain.run_traced(ctx);
        std::cout << "summary   : " << trace["summary"]    << "\n\n";
        std::cout << "translated: " << trace["translated"] << "\n";

        std::cout << "\nMemory stored " << mem->messages().size()
                  << " messages in JSON file.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "chain failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
