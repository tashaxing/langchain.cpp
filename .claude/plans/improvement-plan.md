# langchain.cpp 优化、扩展与完善方案

> 基于对项目代码的全面阅读和分析，从**架构完整性**、**功能覆盖度**、**代码质量**、**测试覆盖**、**工程实践**五个维度整理。
> **更新日期**: 2026-06-08

---

## 当前项目概览

| 维度 | 状态 |
|------|------|
| **模块数** | 19 个核心模块 |
| **测试文件** | 19 个，约 4000 行测试代码 |
| **测试状态** | **18/18 CTest 全部通过** |
| **示例** | 24 个（01-23 + 00_smart_app） |
| **LLM 后端** | 8 个（OpenAI, Anthropic, Gemini, Groq, Ollama, DeepSeek, Qwen, llama.cpp） |
| **CMake 特性** | `LC_ENABLE_OPENSSL`, `LC_ENABLE_LLAMA`, `LC_ENABLE_FAISS` |

---

## 一、与原始 LangChain 对比：还缺少的核心组件

### 1.1 Output Parser（输出解析器）— **高优先级缺失**

原始 LangChain 的 `OutputParser` 是非常核心的组件，用于将 LLM 的原始文本输出解析为结构化数据（JSON、Pydantic 模型、列表、枚举等）。

**当前状态**: 项目中完全没有输出解析器概念。Agent 的 `result.output` 永远是纯字符串。

**建议实现**:
```cpp
// 建议新增 include/output_parser/output_parser.h
class IOutputParser {
public:
    virtual ~IOutputParser() = default;
    virtual json parse(const std::string& text) = 0;
    virtual std::string format_instructions() const = 0;
};

class JsonOutputParser : public IOutputParser { ... };
class ListOutputParser : public IOutputParser { ... };
class EnumOutputParser : public IOutputParser { ... };
```

### 1.2 结构化输出 / JSON Schema 约束 — **高优先级**

当前 `ChatResponse.message.content` 是纯文本。现代 LLM 应用强烈需要结构化输出（OpenAI 的 `response_format: {type: "json_object"}`、function calling 等）。

**建议**: 在 `ChatRequest` 中增加 `response_format` 字段，并在各 LLM 后端中支持。

### 1.3 Document Loaders — **中等优先级**

当前仅有 `TextFileLoader`，非常单薄。

**建议增加**:
- `PDFLoader`（基于 pugixml 或集成 pdfium）
- `CSVLoader`
- `DirectoryLoader`（批量加载目录）
- `WebLoader`（爬取网页）

### 1.4 Chat History / Conversation Buffer Window 的 Token 感知 — **中等优先级**

当前 `WindowMemory` 是基于消息数量的滑动窗口，不是基于 token 数量的。这会导致长消息很快填满上下文窗口。

**建议**: 增加 `TokenBufferMemory`，基于 token 计数（或字节估算）进行滑动窗口裁剪。

### 1.5 Callbacks / Tracing 集成 — **中等优先级**

当前 `HookManager` 提供了基础的生命周期钩子，但缺少与外部追踪系统（LangSmith、Langfuse、OpenTelemetry）的集成。

**建议**: 提供开箱即用的 `LangSmithHook`、`OpenTelemetryHook`。

---

## 二、架构层面的优化建议

| 项目 | 状态 | 说明 |
|------|------|------|
| 2.1 Async LLM API | **✅ 已完成** | `async_invoke()` / `async_invoke_stream()` + 线程池。测试 360 行，示例 `18_async_llm.cpp` |
| 2.2 VectorStore CRUD | **✅ 已完成** | `delete_documents` + `update_document`。InMemory/Sqlite 全实现 + 测试 + 示例。⚠️ Faiss 仍抛 `not implemented` |
| 2.3 本地 Embedding | **⚠️ 部分完成** | Hashing/Http/Ollama/LlamaCpp ✅。缺失: ONNX Runtime |
| 2.4 工具参数校验 | **✅ 已完成** | `ITool::validate()` 虚方法 + `validate_args()` 实现。检查 required + 基本类型。7 个测试用例 |

---

## 三、代码质量与工程实践

### 3.1 测试覆盖 — **✅ 已完成**

全部 18 个 CTest 目标通过：

| 模块 | 测试文件 | 覆盖内容 |
|------|---------|---------|
| PromptTemplate | test_basic, test_rag | ✅ |
| Memory | test_memory.cpp (240) | ✅ Buffer/Window/LongTerm(JSON+SQLite), 多模态 |
| Tool | test_embedding_tool.cpp (~280) | ✅ Calculator, HttpGet, Registry, FunctionTool, **参数校验** |
| Embedding | test_embedding_tool + test_llamacpp_embedding | ✅ Hashing, 配置, LlamaCpp(条件) |
| Async LLM | test_async_llm.cpp (360) | ✅ async_invoke/stream, hooks, 错误传播 |
| Agent | test_agent.cpp (315) | ✅ ReActAgent, ToolCallingAgent, streaming, max_iterations |
| Skill | test_skill.cpp (225) | ✅ Prompt/Retrieval/Chain/Router/Registry |
| API Server/Client | test_api_server + test_api_client | ✅ 路由, OpenAI 端点, health, 客户端 |
| MCP | test_mcp_server.cpp (131) | ✅ |
| A2A | test_a2a.cpp (208) | ✅ Types, 往返, discovery, cancel |
| RAG | test_rag.cpp (183) | ✅ |
| Harness | test_harness.cpp (211) | ✅ 自检, 修正, 进化, null memory |
| Hook | test_hook.cpp (240) | ✅ |
| Config | test_config.cpp (~280) | ✅ Load/get/set/save, **热重载**, **环境变量插值**, **校验规则** |
| Utils | test_utils.cpp (206) | ✅ |
| VectorStore | test_vectorstore.cpp (188) | ✅ InMemory, Sqlite, CRUD, filter |

### 3.2 示例覆盖 — **✅ 已完成**

24 个示例覆盖全部 19 个模块：

| 编号 | 示例 | 覆盖模块 |
|------|------|---------|
| 01 | `01_prompt_basic.cpp` | PromptTemplate |
| 02 | `02_chat_http.cpp` | LLM + BufferMemory |
| 03 | `03_react_agent.cpp` | ReActAgent + LongTermMemory |
| 04 | `04_rag.cpp` | VectorStore(Sqlite) + Embedding |
| 05 | `05_chain_skill.cpp` | ChainSkill + LongTermMemory(JSON) |
| 06 | `06_api_server.cpp` | ApiServer + Hooks + Memory |
| 07 | `07_custom_api.cpp` | ApiServer (custom routes, SSE) |
| 08 | `08_custom_tools_skills.cpp` | Tool + SkillLoader + Agent |
| 09 | `09_persistent_memory.cpp` | Memory (all types) |
| 10 | `10_a2a_agent.cpp` | A2A Server/Client |
| 11 | `11_rag_pipeline.cpp` | Document + TextSplitter + VectorStore + Skill |
| 12 | `12_api_client.cpp` | HttpClient + AIClient + 多模态 |
| 13 | `13_harness_usage.cpp` | HarnessAgent |
| 14 | `14_config_usage.cpp` | Config + Singleton |
| 15 | `15_tool_calling_agent.cpp` | **ToolCallingAgent + 流式事件 + 多模态** |
| 16 | `16_mcp_client_server.cpp` | **McpServer + McpClient** |
| 17 | `17_hooks_observability.cpp` | **HookManager (自定义 IHook, ScopedSpan, 计时)** |
| 18 | `18_async_llm.cpp` | **Async LLM (invoke/stream, 错误, Hook 线程)** |
| 19 | `19_bm25_retriever.cpp` | **BM25Retriever + MultiQueryRetriever** |
| 20 | `20_window_memory.cpp` | **WindowMemory + 会话管理** |
| 21 | `21_text_splitter.cpp` | **Character/Recursive TextSplitter + 自定义长度** |
| 22 | `22_vectorstore_crud.cpp` | **VectorStore CRUD + 元数据过滤 + 持久化** |
| 23 | `23_utilities.cpp` | **Logging, Timer, EventBus, Compress, Strings, FS** |
| 00 | `00_smart_app/` | 综合应用 |

### 3.3 错误处理 — **✅ 已完成**

统一策略已确立并实施：

| 场景 | 策略 | 已实现 |
|------|------|--------|
| 对外 API（查找失败） | `get()` 返回 `nullptr` + `get_or_throw()` 抛 `LCError` | ✅ `ToolRegistry` |
| 对外 API（参数错误） | 抛 `LCError` | ✅ `Config::section()` 等 |
| 内部解析（JSON） | `is_discarded()` 检查 + 抛 `LCError` | ✅ 各 LLM/Embedding 后端 |
| 工具参数校验 | `ValidationResult` 返回（不抛异常，让调用者决定） | ✅ `ITool::validate()` |

### 3.4 线程安全 — **✅ 已完成**

`InMemoryVectorStore` / `BufferMemory` / `WindowMemory` / `HookManager` 均已加锁。

### 3.5 内存管理 — **✅ 已完成**

`SqliteVectorStore` / `FaissVectorStore` 均使用 `std::unique_ptr<Impl>`。

---

## 四、功能扩展建议

| 项目 | 状态 | 说明 |
|------|------|------|
| 4.1 多模态支持 | **⚠️ 部分完成** | `content_parts` 完整链路 ✅。待实现: HttpLLM gpt-4o 格式传递, OCR 工具 |
| 4.2 Agent 流式事件 | **✅ 已完成** | Thought/ToolCall/Observation/Answer/Error/Done。双 Agent 支持 + 测试 + 示例 |
| 4.3 配置系统增强 | **✅ 已完成** | 环境变量插值 `${VAR:-default}` ✅、热重载 `check_reload()` ✅、校验规则 `validate()` ✅ |
| 4.4 日志动态调整 | **⚠️ 部分完成** | `log::set_level()` 已存在 + 示例演示。待完善: 环境变量/配置文件初始化 |

---

## 五、具体可执行的改进清单

### 短期（1-2 周）— 高优先级

1. **增加 Output Parser 模块** — 填补核心能力空白
2. **增加更多 Document Loader** — PDF、CSV、Directory
3. **日志级别环境变量初始化** — `LC_LOG_LEVEL=debug` 自动设置

### 中期（1 个月）— 中等优先级

4. **实现 Token-aware Memory** — 基于字节数或 token 数的滑动窗口裁剪
5. **增加本地 Embedding 支持** — ONNX Runtime 嵌入
6. **实现结构化输出 / JSON Schema 约束** — `ChatRequest::response_format`
7. **增加 LangSmith / OpenTelemetry Hook** — 外部追踪系统集成
8. **FaissVectorStore::update_document 实现**

### 长期（2-3 个月）— 探索性

9. **深度多模态** — 图像理解、OCR 工具、OpenAI 多模态消息格式
10. **分布式 VectorStore 支持** — Milvus、Pinecone、Weaviate 客户端
11. **模型路由与负载均衡** — 在 `ApiServer` 中支持多后端轮询/故障转移
12. **Agent 编排（Multi-Agent）** — Supervisor、GroupChat 模式

---

## 六、当前代码的亮点（保持）

- ✅ **严格的分层架构** — 无循环依赖
- ✅ **NVI 模式 + Hook 系统** — 优雅的观测性设计
- ✅ **PIMPL 隔离 httplib** — 编译隔离做得好
- ✅ **多模态 Message 设计** — `content_parts` 预留扩展性，已打通完整链路
- ✅ **A2A 协议支持** — 走在标准前沿
- ✅ **Harness 自检循环** — 独特的自我进化能力
- ✅ **C++17 兼容** — 没有滥用新特性
- ✅ **测试覆盖全面** — 18/18 测试通过
- ✅ **示例覆盖完整** — 24 个示例覆盖全部 19 个模块
- ✅ **线程安全** — Memory 和 VectorStore 已加锁
- ✅ **Async LLM API** — 基于线程池的异步调用
- ✅ **丰富的 LLM 后端** — 8 个提供商支持
- ✅ **BM25 + MultiQuery Retriever** — 稀疏检索和查询扩展
- ✅ **EventBus + Signal/Slot** — 进程内事件总线
- ✅ **Timer + ThreadPool** — 基础并发设施
- ✅ **Skill 系统** — Prompt/Retrieval/Chain/Router/SkillLoader 完整链路
- ✅ **MCP 协议支持** — Model Context Protocol 服务端和客户端
- ✅ **工具参数校验** — 轻量级 JSON Schema 校验（required + 基本类型）
- ✅ **配置系统增强** — 环境变量插值、热重载、校验规则
- ✅ **错误处理统一** — `get()` / `get_or_throw()` 双模式 + `ValidationResult`
