// examples/11_rag_pipeline.cpp — full RAG pipeline:
// TextFileLoader -> RecursiveCharacterTextSplitter -> InMemoryVectorStore -> RetrievalSkill
#include "langchain.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace langchain;

// Mock LLM for demo purposes: returns a canned response.
class EchoLLM : public llm::ILLM
{
public:
    std::string name() const override
    {
        return "EchoLLM";
    }

protected:
    llm::ChatResponse invoke_impl(const llm::ChatRequest& req) override
    {
        llm::ChatResponse resp;
        resp.message.role = Role::Assistant;
        resp.message.content = "[Mock LLM response based on retrieved context]";
        return resp;
    }
};

// Create a demo text file on disk.
void write_demo_file(const std::string& path)
{
    std::ofstream ofs(path);
    ofs << "C++20 introduces several major features that modernize the language.\n"
        << "Modules (import/export) replace header files and reduce compilation times.\n"
        << "Coroutines enable efficient asynchronous programming with co_await and co_yield.\n"
        << "Concepts provide compile-time constraints on template parameters.\n"
        << "Ranges unify algorithms and views under a single composable framework.\n"
        << "std::format replaces printf and iostream formatting with a Python-like syntax.\n"
        << "std::span is a non-owning view over a contiguous sequence, like string_view for arrays.\n"
        << "std::jthread automatically joins on destruction, preventing thread leaks.\n"
        << "std::source_location captures source code position for debugging and logging.\n"
        << "Three-way comparison (spaceship operator <=>) simplifies operator overloading.\n"
        << "Constexpr improvements allow more code to run at compile time.\n"
        << "Designated initializers bring C-style struct initialization to C++.\n"
        << "std::bit_cast enables safe type punning between equally-sized types.\n"
        << "Calendar and timezone support in <chrono> replaces external date libraries.\n"
        << "std::stop_token and std::stop_source provide cooperative cancellation.\n"
        << "Overall, C++20 makes the language safer, more expressive, and easier to teach.\n";
}

int main()
{
    std::filesystem::create_directories("build");
    const std::string demo_path = "build/rag_pipeline_demo.txt";
    write_demo_file(demo_path);

    // Step 1: Load the document.
    std::cout << "=== Step 1: Load document ===\n";
    auto loader = std::make_shared<document::TextFileLoader>(demo_path);
    auto docs = loader->load();
    std::cout << "Loaded " << docs.size() << " document(s), total chars: "
              << (docs.empty() ? 0 : docs[0].content.size()) << "\n\n";

    // Step 2: Split into chunks.
    std::cout << "=== Step 2: Split into chunks ===\n";
    text_splitter::RecursiveCharacterTextSplitter::Config cfg;
    cfg.chunk_size = 300;
    cfg.chunk_overlap = 50;
    text_splitter::RecursiveCharacterTextSplitter splitter(cfg);

    auto chunks = splitter.split_documents(docs);
    std::cout << "Split into " << chunks.size() << " chunks\n";
    for (std::size_t i = 0; i < chunks.size() && i < 5; ++i)
    {
        std::cout << "  Chunk " << i << " (" << chunks[i].content.size()
                  << " chars): " << chunks[i].content.substr(0, 80) << "...\n";
    }
    std::cout << "\n";

    // Step 3: Embed and store.
    std::cout << "=== Step 3: Embed and store ===\n";
    auto embedder = std::make_shared<embedding::HashingEmbedding>(256);
    auto store = std::make_shared<vectorstore::InMemoryVectorStore>(embedder);
    store->add_documents(std::move(chunks));
    std::cout << "Stored " << store->size() << " documents\n\n";

    // Step 4: Retrieve.
    std::cout << "=== Step 4: Retrieve ===\n";
    auto hits = store->similarity_search("What is std::span?", 2);
    for (const auto& h : hits)
    {
        std::cout << "  [score=" << h.score << "] "
                  << h.doc.content.substr(0, 100) << "...\n";
    }
    std::cout << "\n";

    // Step 5: RAG with RetrievalSkill.
    std::cout << "=== Step 5: RAG with RetrievalSkill ===\n";
    auto llm = std::make_shared<EchoLLM>();
    prompt::PromptTemplate tmpl(
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:");

    skill::RetrievalSkill rag_skill(
        "RAG",
        "Answer questions using retrieved documents.",
        llm,
        store,
        tmpl,
        2,              // top-k
        "question",     // query key
        "context");     // context key

    skill::SkillContext ctx;
    ctx.vars["question"] = "What is std::span?";
    std::string answer = rag_skill.invoke(ctx);
    std::cout << "Answer: " << answer << "\n";

    return 0;
}
