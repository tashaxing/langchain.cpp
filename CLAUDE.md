# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`langchain.cpp` is a pragmatic C++17 agent framework inspired by LangChain, built as a single static library with optional examples, tests, service binaries, and desktop UI clients. It provides LLM abstraction, OpenAI-compatible HTTP client/server APIs, agents, tools, skills, memory, vector stores, MCP, A2A, lifecycle hooks, and a production-style `00_smart_app` example.

The authoritative architecture write-up lives at [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Read it before non-trivial changes.

## Build

CMake is the build system. Linux is the primary runtime target; Windows and macOS are also supported.

### Linux / macOS

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Windows

Use the Visual Studio generator and the `build_x64/` directory:

```bash
cmake -B build_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Release
```

For Debug builds on Windows, MSVC PDB files can occasionally become stale or corrupt. If you see errors such as `program database file has an obsolete format`, delete the affected `.pdb` build artifact and rebuild. If MSBuild reports `out of heap space` or page-file errors, retry with a lower parallelism value and/or increase the system page file.

### Feature flags

- `LC_ENABLE_OPENSSL` (default OFF) — HTTPS support in cpp-httplib. Sets `CPPHTTPLIB_OPENSSL_SUPPORT` on the `lc_httplib` interface target. Do not define it manually in `.cpp` files.
- `LC_ENABLE_LLAMA` (default OFF) — bundled llama.cpp backend, defines `LC_HAS_LLAMA=1`.
- `LC_ENABLE_FAISS` (default OFF) — FAISS-backed vector store, defines `LC_HAS_FAISS=1`.
- `LC_BUILD_FLTK_UI` (default OFF) — builds the optional FLTK desktop UI client for `00_smart_app`. Uses bundled `deps/fltk-release-1.4.5` and `deps/md4c-release-0.5.3`.
- `LC_BUILD_EXAMPLES` (default ON) — builds examples.
- `LC_BUILD_TESTS` (default ON) — builds tests.
- `LC_INSTALL` (default ON) — generates install rules and CMake package exports.

All dependencies are vendored under `deps/`; nothing is fetched at build time except the optional OpenSSL `ExternalProject_Add` step.

## Tests

Two test styles coexist:

- `tests/test_basic.cpp` — assert-based smoke test, no framework dependency.
- Other `tests/test_*.cpp` files — GoogleTest-based, built when `deps/googletest-v1.17.0/` is present.

```bash
# Linux / macOS
cmake --build build --config Release
ctest --test-dir build --output-on-failure

# Windows
cmake --build build_x64 --config Debug
ctest --test-dir build_x64 -C Debug --output-on-failure

# Single test
ctest --test-dir build -R test_hook --output-on-failure
```

### Test file placement rules

- All test code must live under `tests/`.
- Test file names should be clean and framework-neutral, e.g. `test_memory.cpp`, not `test_gtest_memory.cpp`.

## Examples and applications

### Single-file examples

`examples/*.cpp` are standalone demos. Each becomes its own executable and should be run from the repository root. Network examples read environment variables such as `LC_BASE_URL`, `LC_API_KEY`, and `LC_MODEL`.

### 00_smart_app

`examples/00_smart_app/` is the production-style example application. It builds a self-contained OpenAI-compatible agent service with:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /v1/memory`
- `POST /v1/memory/clear`
- `GET /v1/config`
- `POST /v1/config/reload`
- `POST /v1/rag/ingest`

It supports streaming/non-streaming chat, ToolCallingAgent routing, skills, tools, SQLite memory, RAG ingestion, and application-level XML config.

Windows/Linux start/stop scripts kill by process name rather than PID files. Keep that behavior unless explicitly asked otherwise.

### FLTK desktop chat client

When `LC_BUILD_FLTK_UI=ON`, `examples/00_smart_app/ui/` builds `00_smart_app_chat`, a lightweight FLTK desktop client for the remote `00_smart_app` service.

Important constraints:

- Client configuration is local to `config/chat_client_config.xml`; do not read server-side `app_config.xml` for client UI options.
- Model list, default model, host, port, session id, stream flag, temperature, top_p, top_k, max_tokens, timeouts, and console logging are all client-side config values.
- The Windows client should not pop a console window by default. The target is configured as a Windows GUI subsystem in CMake while keeping portable `main()` via `/ENTRY:mainCRTStartup` on MSVC.
- The chat view is a custom FLTK bubble view with Markdown rendering via md4c. It supports common AI-chat Markdown including headings, bold, italic, inline code, code blocks, lists, task lists, quotes, links, tables, strikethrough, underline, wiki links, image placeholders, and LaTeX placeholders.
- Stream rendering is throttled and batched to avoid scrollbar flicker. Avoid per-token full relayouts.

## API performance logging

`ApiServer` emits structured performance logs with grep-friendly markers:

- `[API_PERF][REQUEST]` — request start
- `[API_PERF][RESPONSE]` — response completion

The response helper is named `log_api_response_end`. Keep request/response naming explicit. These logs include call id, route, method/status, model, stream flag, request/response bytes, elapsed microseconds/milliseconds, session id, and remote address where available.

For streaming responses, final latency is logged when the streaming provider completes.

## Architecture invariants

These rules are easy to break without realizing it. The architecture doc has the full picture; this is the short list:

- Strict bottom-up layering: `util → prompt → hook → llm → embedding → vectorstore → memory → tool → mcp → agent → skill → api → a2a`.
- `ILLM` uses the NVI pattern. Backends override `invoke_impl()` / `invoke_stream_impl()`, never public `invoke()` / `invoke_stream()`.
- Components without explicit `set_hooks()` should fall back to `hook::HookManager::global()`.
- `ApiServer` and `McpServer` hide cpp-httplib behind PIMPL to avoid leaking `httplib.h` through public headers.
- `httplib::Server` shutdown can be racy. Follow the existing `stop()` / worker join patterns.
- Custom `ApiServer` GET routes are dispatched via `pre_routing_handler`; body-carrying methods use generic `svr.Post(".*")`, `svr.Put(".*")`, `svr.Patch(".*")`, and `svr.Delete(".*")` handlers after explicit routes.

## Coding conventions

Hard requirements:

- Google C++ style with Allman braces.
- Header/implementation separation is mandatory for non-template code.
- C++17 is the ceiling.
- Cross-platform support: Linux, Windows, and macOS.
- Prefer standard C++ facilities (`std::filesystem`, `std::thread`, `<chrono>`) over platform APIs.
- Platform-specific behavior should be handled in CMake when possible; if source-level platform code is unavoidable, guard it carefully.
- Use `/` in source path strings; avoid hard-coded `\\`.
- Source encoding is UTF-8; comments and documentation must be English.
- Match surrounding style and comment density.
- Do not refactor unrelated code while solving a targeted task.

## Module layout

- `include/<module>/` — public headers.
- `src/<module>/` — implementations compiled into the `langchain` static library.
- `examples/` — single-file demos and `00_smart_app`.
- `tests/` — test suite.
- `deps/` — vendored third-party dependencies.
- `docs/` — architecture and design documents.

Adding a new library module means adding public headers under `include/`, implementation files under `src/`, and updating `include/langchain.h` if the module is part of the public umbrella API. CMake picks up new `.cpp` files after reconfigure via `CONFIGURE_DEPENDS`.

## Behavioral guidelines

Bias toward simple, verifiable changes.

- State assumptions before coding when requirements are ambiguous.
- Prefer the smallest change that solves the task.
- Do not add speculative abstractions.
- Touch only files required by the request.
- Verify with build/tests or explain exactly why verification was blocked.
- Report failures honestly, including compiler or runtime output.
