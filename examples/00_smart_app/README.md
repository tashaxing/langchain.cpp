# 00_smart_app — Comprehensive AI Smart Agent Application

A production-ready smart agent application built on **langchain.cpp**, demonstrating the full capabilities of the framework as a library.

## Features

- **OpenAI-compatible HTTP API** — `/v1/models`, `/v1/chat/completions`
- **Streaming & Non-streaming** — SSE streaming for real-time responses
- **Multimodal** — Accepts text + image_url/image_base64 in messages
- **ToolCallingAgent** — Agent with tools, skills, and persistent memory
- **RAG Pipeline** — Document ingestion, chunking, embedding, and retrieval
- **Persistent Memory** — Per-session SQLite-backed conversation history
- **Configuration-driven** — XML config with hot-reload support
- **Cross-platform** — CMake-based, works on Windows, Linux, macOS

## Deployment Directory Structure

After build or install, the application uses a self-contained directory layout:

```
00_smart_app/
├── bin/
│   ├── 00_smart_app       # Executable
│   ├── start.sh           # Linux/macOS start script
│   ├── stop.sh            # Linux/macOS stop script
│   ├── start.bat          # Windows start script
│   └── stop.bat           # Windows stop script
├── config/
│   └── app_config.xml     # Application configuration
├── data/
│   └── knowledge/         # RAG knowledge base documents
├── log/                   # Runtime logs (created automatically)
├── lib/                   # Optional shared libraries
└── skills/                # SKILL.md skill definitions
```

All paths in `config/app_config.xml` are relative and resolved at runtime against
the application base directory (the parent of `bin/`). This means you can copy the
entire directory to any location on the same machine and it will work without
modification.

## Quick Start

### Build

**Linux / macOS:**

```bash
cd /path/to/langchain.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Windows:**

```bash
cd /path/to/langchain.cpp
cmake -B build_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Release
```

### Run (Development)

From the build directory:

```bash
# Linux / macOS
cd build/examples/00_smart_app
./bin/00_smart_app

# Windows
cd build_x64\examples\00_smart_app
bin\00_smart_app.exe
```

The executable automatically finds `config/app_config.xml` relative to its location.

### Run (Production)

Use the provided start/stop scripts:

```bash
# Linux/macOS
cd build/examples/00_smart_app
./bin/start.sh
./bin/stop.sh

# Windows
cd build_x64\examples\00_smart_app
bin\start.bat
bin\stop.bat
```

### Desktop Chat Client (optional FLTK UI)

`00_smart_app` also provides an optional lightweight FLTK desktop chat client.
The client is a standalone executable that can run on Windows, macOS, or Ubuntu
and connect to a `00_smart_app` server running on another machine.

Build with FLTK UI enabled:

```bash
# Linux / macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLC_BUILD_FLTK_UI=ON
cmake --build build --target 00_smart_app_chat -j

# Windows
cmake -B build_x64 -G "Visual Studio 17 2022" -A x64 -DLC_BUILD_FLTK_UI=ON
cmake --build build_x64 --config Release --target 00_smart_app_chat
```

Configure all desktop client options in `config/chat_client_config.xml`:

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

Launch the client:

```bash
# Uses config/chat_client_config.xml by default
./bin/00_smart_app_chat

# Override host/port temporarily from CLI
./bin/00_smart_app_chat --host 192.168.1.50 --port 8080 --session alice-desktop
```

Supported options:

```text
--config <path>            Local client config XML
--host <ip-or-host>        Backend agent server host
--port <port>              Backend agent server port
--model <id>               Initial model id
--session <id>             Session id for X-Session-Id, default desktop
--console                  Enable console logs
--no-console               Disable console logs
--no-stream                Use non-streaming chat
--temperature <float>      Sampling temperature, default 0.7
--top-p <float>            Nucleus sampling top_p, default 0.95
--top-k <int>              Top-k sampling value, default 50
--max-tokens <int>         Max completion tokens, default 1024
--connect-timeout <sec>    Connection timeout, default 10
--read-timeout <sec>       Read timeout, default 300
```

The desktop client calls the already-running `00_smart_app` agent service. It does
not configure or require an API key because the backend service already encapsulates
LLM access and exposes a pure internal agent API.

For HTTPS endpoints, build with `LC_ENABLE_OPENSSL=ON` as well. On Ubuntu
desktop, FLTK may require X11 development packages such as `libx11-dev`,
`libxext-dev`, `libxft-dev`, `libxinerama-dev`, `libxcursor-dev`,
`libxrender-dev`, and `libxfixes-dev`.

### Install

```bash
# Linux / macOS
cmake --install build --prefix /opt/00_smart_app

# Windows
cmake --install build_x64 --prefix C:\00_smart_app
```

After install, the directory structure is fully self-contained:

```bash
/opt/00_smart_app/bin/start.sh
/opt/00_smart_app/bin/stop.sh
```

You can copy `/opt/00_smart_app` to any other machine with the same OS and it will
run without any path changes.

### Test

```bash
# Health check
curl http://localhost:8080/healthz

# List models
curl http://localhost:8080/v1/models

# Chat (non-streaming)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"saas-kimi-k25","messages":[{"role":"user","content":"hello"}]}'

# Chat (streaming)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"saas-kimi-k25","messages":[{"role":"user","content":"hello"}],"stream":true}'

# Memory inspection
curl http://localhost:8080/v1/memory -H "X-Session-Id: test-session"

# Clear memory
curl -X POST http://localhost:8080/v1/memory/clear -H "X-Session-Id: test-session"

# Config reload
curl -X POST http://localhost:8080/v1/config/reload

# RAG ingest
curl -X POST http://localhost:8080/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"dir":"data/knowledge"}'
```

## Configuration

Edit `config/app_config.xml` to customize:

| Section | Description |
|---------|-------------|
| `app` | Logging level, log directory, worker threads |
| `llm` | Base URL, API key, default model, temperature, max_tokens |
| `api_server` | Host, port, API key, timeouts |
| `embedding` | Embedding provider, model, dimensions |
| `vectorstore` | SQLite database path (relative to app base) |
| `memory` | SQLite database path (relative to app base), default session ID |
| `rag` | Knowledge directory (relative to app base), chunk size, top-k |
| `agent` | Agent type, max iterations, system prompt |
| `skill` | Skills directory (relative to app base) for SKILL.md loading |
| `tool` | Enable/disable individual tools |

All paths in the config file are relative to the application base directory.
For example, `log_dir=log` resolves to `<app_base>/log/`, and
`db_path=data/memory.db` resolves to `<app_base>/data/memory.db`.

## Architecture

```
00_smart_app/
├── main.cpp         # Entry point, signal handling, main loop
├── app_config.cpp   # Config loading and accessors
├── app_paths.cpp    # Cross-platform executable path resolution
├── app_tools.cpp    # Tool registry (calculator, http_get, datetime, echo)
├── app_hooks.cpp    # Lifecycle hook registration
├── app_rag.cpp      # Vector store, document ingestion, retrieval skill
├── app_agent.cpp    # ToolCallingAgent factory
├── app_server.cpp   # ApiServer setup, LLM registration, custom routes
└── config/
    └── app_config.xml
```

## Custom Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Health check |
| GET | `/v1/models` | List registered models |
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| GET | `/v1/memory` | List session messages |
| POST | `/v1/memory/clear` | Clear session memory |
| GET | `/v1/config` | View current configuration |
| POST | `/v1/config/reload` | Hot-reload configuration from disk |
| POST | `/v1/rag/ingest` | Ingest documents into vector store |

## Environment Variables

- `SMART_APP_DIR` — Override the application directory for scripts

## License

Same as langchain.cpp.
