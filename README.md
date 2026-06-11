# langchain.cpp

> A pragmatic C++17 framework for building LLM applications, agents, tools, RAG pipelines, OpenAI-compatible services, and lightweight desktop chat clients.

`langchain.cpp` brings the core ideas of LangChain to modern C++ without turning every concept into a heavyweight abstraction. It ships as a single static library, uses vendored dependencies, builds with CMake, and runs on Linux, Windows, and macOS.

- OpenAI-compatible HTTP client and server
- Streaming and non-streaming chat completions
- ReAct and provider-native tool-calling agents
- Tools, skills, memory, vector stores, and RAG
- MCP client/server support
- A2A agent-to-agent protocol support
- Lifecycle hooks for observability and middleware
- Production-style `00_smart_app` service example
- Optional FLTK desktop chat client with Markdown bubble rendering

## Why langchain.cpp?

Most LLM frameworks are Python-first. That is fine until you need:

- a native C++ service runtime,
- offline or embedded deployment,
- static linking,
- low-level control over memory and threading,
- integration with existing C++ infrastructure,
- or a small framework that can be read and debugged directly.

`langchain.cpp` is designed for those cases.

## Features

| Area | What you get |
|---|---|
| LLMs | `ILLM` abstraction, HTTP/OpenAI-compatible backend, optional llama.cpp backend |
| API client | Generic REST client plus OpenAI-compatible chat helpers and SSE streaming |
| API server | `/v1/models`, `/v1/chat/completions`, streaming, multimodal pass-through, custom routes |
| Agents | `ReActAgent` and `ToolCallingAgent` |
| Tools | Built-in tools, custom tools, tool registry |
| Skills | Prompt, retrieval, chain, router, and skill registry primitives |
| RAG | Text splitting, embedding, vector stores, retrievers, retrieval skills |
| Memory | Window memory and SQLite-backed long-term memory |
| MCP | Model Context Protocol client/server over JSON-RPC |
| A2A | Google A2A-compatible agent card, task lifecycle, send/subscribe APIs |
| Hooks | Before/After lifecycle events for LLM, agent, tool, skill, and API calls |
| UI | Optional FLTK desktop chat client for `00_smart_app` |

## Quick start

### Requirements

- CMake 3.15+
- C++17 compiler
  - GCC / Clang on Linux
  - Apple Clang on macOS
  - MSVC with Visual Studio 2022 on Windows

All regular dependencies are vendored under `deps/`. No package manager is required for the default build.

### Build

#### Linux / macOS

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### Windows

```bash
cmake -B build_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Release
```

### Optional HTTPS support

HTTPS support for public providers such as OpenAI requires OpenSSL:

```bash
cmake -B build -DLC_ENABLE_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`LC_ENABLE_OPENSSL=ON` builds bundled OpenSSL 3.3.6 via `ExternalProject`. You need `perl` on `PATH`.

## Hello, LLM

```cpp
#include "langchain.h"

#include <cstdlib>
#include <iostream>
#include <memory>

int main()
{
    langchain::llm::HttpLLMConfig cfg;
    cfg.base_url = "http://localhost:11434";
    cfg.model = "qwen2.5:0.5b";

    auto llm = std::make_shared<langchain::llm::HttpLLM>(cfg);
    std::cout << llm->complete("Explain RAII in one sentence.") << "\n";
}
```

For HTTPS providers:

```cpp
cfg.base_url = "https://api.openai.com";
cfg.api_key = std::getenv("OPENAI_API_KEY");
cfg.model = "gpt-4o-mini";
```

## Tool-calling agent

```cpp
auto llm = std::make_shared<langchain::llm::HttpLLM>(cfg);

langchain::tool::ToolRegistry tools;
tools.add(langchain::tool::make_calculator_tool());

langchain::agent::ToolCallingAgent agent(llm, std::move(tools));
auto result = agent.run("What is (17 * 23) + 9?");

std::cout << result.output << "\n";
for (const auto& step : result.steps)
{
    std::cout << "tool=" << step.tool_name
              << " input=" << step.tool_input
              << " observation=" << step.observation << "\n";
}
```

## RAG in a few lines

```cpp
auto embedder = std::make_shared<langchain::embedding::HashingEmbedding>(256);
auto store = std::make_shared<langchain::vectorstore::SqliteVectorStore>(embedder, "kb.db");

store->add_documents({
    {"", "std::span is a non-owning view over contiguous memory.", {}, {}},
    {"", "std::jthread joins automatically on destruction.", {}, {}},
});

langchain::skill::RetrievalSkill rag(
    "qa",
    "Answer from the local knowledge base.",
    llm,
    store,
    langchain::prompt::PromptTemplate("Context:\n{context}\n\nQuestion: {question}"),
    3);

langchain::skill::SkillContext ctx;
ctx.vars["question"] = "What does std::span do?";
std::cout << rag.run(ctx) << "\n";
```

## Serve an OpenAI-compatible API

```cpp
langchain::api::ApiConfig cfg;
cfg.host = "0.0.0.0";
cfg.port = 8080;

langchain::api::ApiServer server(cfg);
server.register_model("local", llm);
server.start();

// ... later
server.stop();
```

Then call it like any OpenAI-compatible endpoint:

```bash
curl http://localhost:8080/v1/models

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}]}'
```

Streaming:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## `00_smart_app`: full agent service example

`examples/00_smart_app` is a production-style application built with the framework. It demonstrates:

- OpenAI-compatible HTTP API
- multiple model registrations
- ToolCallingAgent integration
- tools and skills
- persistent SQLite memory
- RAG document ingestion
- configuration reload
- performance logging
- optional desktop chat UI

### Build and run

```bash
# Linux / macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 00_smart_app -j
./build/examples/00_smart_app/bin/00_smart_app

# Windows
cmake -B build_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Release --target 00_smart_app
build_x64\examples\00_smart_app\bin\00_smart_app.exe
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | health check |
| GET | `/v1/models` | model list |
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| GET | `/v1/memory` | inspect session memory |
| POST | `/v1/memory/clear` | clear session memory |
| GET | `/v1/config` | inspect config |
| POST | `/v1/config/reload` | hot-reload config |
| POST | `/v1/rag/ingest` | ingest documents into RAG store |

### API performance logs

`ApiServer` emits request/response performance logs with stable markers:

```text
[API_PERF][REQUEST]  ...
[API_PERF][RESPONSE] ...
```

Fields include call id, route, status, elapsed time in milliseconds and microseconds, request/response bytes, model, stream flag, session id, and remote address. These markers are intended for load-test parsing and latency aggregation.

## Optional FLTK desktop chat client

Build with:

```bash
cmake -B build_x64 -G "Visual Studio 17 2022" -A x64 -DLC_BUILD_FLTK_UI=ON
cmake --build build_x64 --config Release --target 00_smart_app_chat
```

Run:

```bash
build_x64\examples\00_smart_app\bin\00_smart_app_chat.exe
```

The client reads its own local config from:

```text
examples/00_smart_app/config/chat_client_config.xml
```

Example:

```xml
<chat_client>
    <server>
        <host>127.0.0.1</host>
        <port>8080</port>
        <connect_timeout_sec>10</connect_timeout_sec>
        <read_timeout_sec>300</read_timeout_sec>
    </server>
    <models default="saas-kimi-k25">
        <model>local-deepseek-v32</model>
        <model>saas-qwen35-397b</model>
        <model>saas-kimi-k25</model>
        <model>saas-deepseek-v32</model>
        <model>saas-glm-5</model>
    </models>
    <ui>
        <console>false</console>
    </ui>
    <chat>
        <session_id>desktop</session_id>
        <stream>true</stream>
        <temperature>0.7</temperature>
        <top_p>0.95</top_p>
        <top_k>50</top_k>
        <max_tokens>1024</max_tokens>
    </chat>
</chat_client>
```

The UI includes:

- WeChat-style chat bubbles
- right-aligned user messages
- left-aligned assistant messages
- streaming rendering with throttled refresh
- Markdown rendering via md4c
- tables, lists, task lists, code blocks, inline code, bold, italic, links, quotes, image placeholders, and math placeholders
- local multi-conversation sidebar
- no Windows console window by default

## Examples

Each `examples/*.cpp` file builds into a standalone executable.

| Example | Description |
|---|---|
| `01_prompt_basic.cpp` | PromptTemplate basics |
| `02_chat_http.cpp` | HTTP chat client |
| `03_react_agent.cpp` | ReAct agent with calculator tool |
| `04_rag.cpp` | RAG with SQLite vector store |
| `05_chain_skill.cpp` | Sequential skill composition |
| `06_api_server.cpp` | OpenAI-compatible API server |
| `07_custom_api.cpp` | Custom REST routes |
| `08_custom_tools_skills.cpp` | Custom tools and skills |
| `09_persistent_memory.cpp` | SQLite long-term memory |
| `10_a2a_agent.cpp` | A2A agent server and client |
| `11_rag_pipeline.cpp` | RAG pipeline components |
| `12_api_client.cpp` | API client helpers |
| `13_harness_usage.cpp` | Evaluation harness usage |
| `14_config_usage.cpp` | YAML config system |
| `15_tool_calling_agent.cpp` | Native tool-calling agent |
| `16_mcp_client_server.cpp` | MCP client/server |
| `17_hooks_observability.cpp` | Lifecycle hooks |
| `18_async_llm.cpp` | Async LLM calls |
| `19_bm25_retriever.cpp` | BM25 retrieval |
| `20_window_memory.cpp` | Windowed conversation memory |
| `21_text_splitter.cpp` | Text splitting |
| `22_vectorstore_crud.cpp` | Vector store CRUD |
| `23_utilities.cpp` | Utility helpers |
| `24_multi_agent_workflow.cpp` | Multi-agent workflow |

Network examples use:

```bash
export LC_BASE_URL=http://localhost:11434
export LC_MODEL=qwen2.5:0.5b
export LC_API_KEY=optional
```

## CMake options

| Option | Default | Description |
|---|---:|---|
| `LC_ENABLE_OPENSSL` | OFF | HTTPS support through bundled OpenSSL |
| `LC_ENABLE_LLAMA` | OFF | llama.cpp backend |
| `LC_ENABLE_FAISS` | OFF | FAISS vector store |
| `LC_BUILD_FLTK_UI` | OFF | FLTK desktop chat client for `00_smart_app` |
| `LC_BUILD_EXAMPLES` | ON | Build examples |
| `LC_BUILD_TESTS` | ON | Build tests |
| `LC_INSTALL` | ON | Install rules and CMake package exports |

Do not define `CPPHTTPLIB_OPENSSL_SUPPORT` manually in source files. Let `LC_ENABLE_OPENSSL` configure it.

## Testing

```bash
# Linux / macOS
cmake --build build
ctest --test-dir build --output-on-failure

# Windows
cmake --build build_x64 --config Debug
ctest --test-dir build_x64 -C Debug --output-on-failure

# Single test
ctest --test-dir build -R test_hook --output-on-failure
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

Layering is intentionally strict:

```text
util → prompt → hook → llm → embedding → vectorstore → memory
                                                          ↓
                                                       tool → mcp → agent → skill → api
                                                                                  ↓
                                                                                a2a
```

No upward dependencies. Keep public headers lightweight and avoid leaking transport-specific headers such as `httplib.h` through public APIs.

## Consuming from CMake

```cmake
find_package(langchain CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE langchain::langchain)
```

## Vendored dependencies

| Dependency | Version | Notes |
|---|---:|---|
| nlohmann/json | 3.12.0 | header-only JSON |
| cpp-httplib | 0.46 | HTTP client/server |
| spdlog | 1.13 | logging |
| pugixml | 1.15 | XML config |
| sqlite | 3.51 | persistence |
| zlib | 1.3.2 | compression |
| yaml-cpp | 0.9.0 | YAML config |
| md4c | 0.5.3 | Markdown parsing for FLTK chat UI |
| FLTK | 1.4.5 | optional desktop UI |
| OpenSSL | 3.3.6 | optional HTTPS |
| GoogleTest | 1.17 | tests |
| llama.cpp | b5018 | optional local inference |
| FAISS | 1.14.2 | optional vector search |

## Design non-goals

- Python compatibility or LCEL-style operator DSL
- async-everywhere API surface
- dozens of provider adapters by default
- hidden global runtime dependencies
- mandatory desktop UI dependencies for library users

## License

TBD.
