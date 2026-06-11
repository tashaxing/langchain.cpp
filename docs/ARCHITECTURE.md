# langchain.cpp — Architecture

A C++17 reimagining of LangChain. Same conceptual layering, smaller surface area, no Python-style "every concept gets a class" tax.

## Layering (bottom → top)

```
util           — Message, Document, json alias, logging, strings, fs, time,
                 timer, compress (zlib), eventbus (Signal + EventBus),
                 singleton
prompt         — PromptTemplate, ChatPromptTemplate
hook           — IHook, FunctionHook, HookManager, ScopedSpan
                 (middleware-style Before*/After* observers)
llm            — ILLM (NVI base — public invoke()/invoke_stream() fire hooks,
                 subclasses override invoke_impl()/invoke_stream_impl()),
                 HttpLLM (OpenAI-compat), LlamaCppLLM (optional, gated by LC_ENABLE_LLAMA)
embedding      — IEmbedding, HttpEmbedding, HashingEmbedding (offline)
vectorstore    — IVectorStore, InMemoryVectorStore, SqliteVectorStore
memory         — BufferMemory, WindowMemory, LongTermMemory (JSON/SQLite backends)
tool           — ITool, FunctionTool, ToolRegistry, calculator/http_get
mcp            — McpClient (JSON-RPC over HTTP, wraps remote tools as ITool)
                 McpServer (exposes a local ToolRegistry to remote peers)
agent          — ReActAgent (text protocol), ToolCallingAgent (OpenAI-style)
skill          — PromptSkill, RetrievalSkill (RAG), ChainSkill (sequential),
                 RouterSkill (branching)
api            — ApiServer (OpenAI-compatible REST: /v1/models,
                 /v1/chat/completions with SSE streaming, /healthz)
a2a            — A2AClient, A2AServer (Google A2A protocol for agent discovery
                 and task exchange)
```

Each module only depends on modules below it on this list — there are no upward edges. `hook/` is referenced by every component that fires lifecycle events (`llm`, `agent`, `skill`, `tool` callers, `mcp` server, `api`, `a2a`) but its only own dependencies are `llm/llm.h` (for typed context pointers) and `util/`.

## Interaction map

```
                    HTTP client (OpenAI-compatible)
                              │
                              ▼
                      ┌──────────────┐
                      │  ApiServer   │  ── /v1/chat/completions, /v1/models
                      └──────┬───────┘     (+ SSE streaming)
                             │ dispatch by model id
                             ▼
                          ILLM ◄─── HookManager (Before/AfterLLM)

        user input
             │
             ▼
      ┌──────────────┐
      │   Skill /    │  ── high-level "do one thing"
      │   Agent      │  ── tool-using loop
      └─────┬────────┘ ◄── HookManager (Before/AfterSkill, Before/AfterAgent)
            │ uses
   ┌────────┼────────────────────┬────────────────┐
   ▼        ▼                    ▼                ▼
 PromptT.  ILLM ── HttpLLM     ToolRegistry    Memory
                   LlamaCppLLM    │  ◄── HookManager (Before/AfterTool)
                                  ▼
                            ITool / MCP tool
                                  │
                                  ▼
                          (vectorstore, http, math, …)

         Remote MCP peer
              │  JSON-RPC
              ▼
         McpServer  ── re-exposes a local ToolRegistry
```

### Lifecycle hooks

Every public entry point on `ILLM`, `ReActAgent`, `ToolCallingAgent`, `ISkill`, and `McpServer::tools/call` fires `Before*` / `After*` events through a `HookManager`. Hooks receive a `HookContext` carrying typed pointers (`mutable_request`, `response`, `agent_input`, `tool_arguments`, …); `Before*` hooks may mutate the request in place, and `After*` hooks see the response plus a wall-clock elapsed time.

Components without an explicit `set_hooks()` fall back to `hook::HookManager::global()`, so process-wide observability can be wired in one place.

The NVI pattern in `ILLM` is load-bearing: backends override `chat_impl()` / `chat_stream_impl()`, never the public `chat()` / `chat_stream()`. That keeps Before/After firing uniform across every backend.

## Dependency choices

| Capability                  | Library (in `deps/`)        | Notes |
|-----------------------------|-----------------------------|-------|
| JSON                        | nlohmann/json 3.12.0        | header-only |
| HTTP client + server (+SSE) | cpp-httplib 0.46            | header-only, HTTPS opt-in |
| Logging                     | spdlog 1.13                 | header-only mode (`FMT_HEADER_ONLY`) |
| Vector persistence          | sqlite 3.51                 | built as static lib |
| XML (skill/config import)   | pugixml 1.15                | static lib |
| Compression                 | zlib 1.3.2                  | bundled, built static |
| HTTPS (optional)            | OpenSSL 3.3.6               | gated by `-DLC_ENABLE_OPENSSL=ON`, built from bundled source |
| Local LLM (optional)        | llama.cpp b5018             | gated by `-DLC_ENABLE_LLAMA=ON` |
| Test framework (optional)   | GoogleTest 1.17             | only used by `tests/test_gtest_*.cpp` |

### Optional feature flags

- `LC_ENABLE_OPENSSL=ON` — required for HTTPS endpoints (api.openai.com, etc.). Triggers a from-source OpenSSL build via `ExternalProject` (needs perl on PATH). Sets `CPPHTTPLIB_OPENSSL_SUPPORT` on the `lc_httplib` interface target — **never define this macro manually in a `.cpp` file**.
- `LC_ENABLE_LLAMA=ON` — pulls in bundled `llama.cpp`, builds `llm/llama_cpp_llm.cpp`, defines `LC_HAS_LLAMA=1` on the public library target.
- `LC_BUILD_EXAMPLES=ON` (default) — the `examples/*.cpp` programs.
- `LC_BUILD_TESTS=ON` (default) — the smoke test (`test_basic.cpp`) plus the GoogleTest suite when `deps/googletest-v1.17.0/` is present.
- `LC_INSTALL=ON` (default) — emits `langchainConfig.cmake` so downstream projects can `find_package(langchain)`.

### Dependencies you may want to add later

- **hnswlib** — header-only ANN, drop-in when the in-memory brute-force store outgrows ~10⁵ docs.
- **tokenizers-cpp** — accurate token counting; current code is byte-based.
- **fmt** — only if you find yourself wanting `std::format`-style output without bumping the C++ standard.

## Build

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# HTTPS to public providers
cmake -B build -DLC_ENABLE_OPENSSL=ON -G "Visual Studio 17 2022" -A x64

# Local llama.cpp backend
cmake -B build -DLC_ENABLE_LLAMA=ON -G "Visual Studio 17 2022" -A x64
```

## Quick start

```cpp
#include "langchain.h"
using namespace langchain;

llm::HttpLLMConfig cfg;
cfg.base_url = "https://api.openai.com";
cfg.api_key  = std::getenv("OPENAI_API_KEY");
cfg.model    = "gpt-4o-mini";
auto llm = std::make_shared<llm::HttpLLM>(cfg);

tool::ToolRegistry tools;
tools.add(tool::make_calculator_tool());

agent::ToolCallingAgent ag(llm, std::move(tools));
auto result = ag.run("What is 17 * 23 + 9?");
std::cout << result.output << "\n";
```

### Serve OpenAI-compatible HTTP

```cpp
api::ApiServer server({.port = 8080});
server.register_model("gpt-4o-mini", llm);
server.start();   // background thread
// ... server.stop() when done
```

### Observe everything

```cpp
hook::HookManager::global().add(
    "trace",
    [](hook::HookContext& ctx) {
        std::cout << hook::to_string(ctx.phase)
                  << " " << ctx.component
                  << " " << ctx.elapsed.count() << "us\n";
    });
```

### A2A agent-to-agent communication

```cpp
// Expose an agent
a2a::AgentCard card;
card.name = "echo-agent";
card.url  = "http://localhost:9099";

a2a::A2AServer a2a_server(card,
    [](const a2a::Task& task, auto update) -> a2a::Task
    {
        a2a::Task result = task;
        result.status.state = a2a::TaskState::Completed;
        return result;
    });
a2a_server.start(9099);

// Talk to a remote agent
a2a::A2AClient client("http://localhost:9099");
a2a::AgentCard remote = client.discover();
a2a::Task task = /* ... */;
a2a::Task result = client.send_task(task);
```

## What is intentionally NOT in this port

- Streaming via coroutines (we use callbacks — `StreamCallback` returns `bool` to abort).
- Pydantic-style runtime validation of tool schemas (the JSON schema is shipped to the LLM but not enforced locally).
- Hundreds of LLM/vector-store adapters — start from `HttpLLM` and `InMemoryVectorStore`; add what you need.
- LCEL pipe operator overloading. Compose by writing C++ functions; readability beats cleverness.
- Async/futures everywhere — hooks fire synchronously, agents loop synchronously. Use `util::Signal` / `util::EventBus` if you need fan-out.
