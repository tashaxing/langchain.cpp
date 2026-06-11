# Plan: 多模态图片持久化 — Memory 端到端支持

## Context

当前 `Message::content_parts`（多模态内容）在 Memory 持久化时被完全丢弃，Agent 传递用户输入时也只传字符串。需要在整个链路中让多模态（图片 base64）随文本一起保存和恢复。

## 设计决策（已确认）

| 问题 | 方案 |
|------|------|
| Schema 迁移 | `ALTER TABLE` 加列（不破坏旧数据） |
| 存储格式 | 整条 Message 序列化为 JSON 存一列（`payload TEXT`） |
| Agent 接口 | 新增 `invoke(const Message&)` 重载（不破坏现有调用） |
| API 返回 | OpenAI 风格 `content` 数组（与入参对称） |

## 修改清单

### 1. [src/memory/memory.cpp](src/memory/memory.cpp) — Memory 存储层

**SQLite schema 迁移** — 在 `open_sqlite_()` 的 `CREATE TABLE` 后加 `ALTER TABLE ADD COLUMN`（try-ignore）：

```sql
ALTER TABLE messages ADD COLUMN payload TEXT;  -- 完整 Message JSON（含 content_parts/tool_calls）
```

**`encode_message()`** — 新增 `content_parts` 序列化（JSON array），返回结果直接作为 payload。

**`decode_message()`** — 优先从 `payload` 反序列化完整 Message（含 content_parts）；若不存在则从旧列构造。

**`Impl::add()` SQLite** — INSERT 增加 `payload` 字段绑定。

**`Impl::messages()` SQLite** — SELECT 增加 `payload` 列，读取时优先使用。

**JSON file 后端** — `append_json_()` 和 `rewrite_json_()` 的 JSON 行中已经含整条 Message JSON，只需补充 `content_parts` 序列化即可。

### 2. [src/agent/agent.cpp](src/agent/agent.cpp) — Agent 新增 Message 重载

**`ReActAgent`** — 新增 `invoke(const Message& user_msg, ...)` 重载：
- 用 `user_msg.content` 作为文本 fallback
- 用 `user_msg.content_parts` 构造 conv 中的 user Message（而不是只传字符串）
- 写 memory 时调用 `mem_->add(Message::user(完整的带 content_parts))` 而不是 `add_exchange(string)`

**`ToolCallingAgent`** — 同上。

### 3. [src/agent/agent.h](src/agent/agent.h) — Agent 头文件

新增 `invoke(const Message& user_msg, ...)` 和 `invoke_stream(const Message& user_msg, ...)` 声明。

### 4. [src/api/server.cpp](src/api/server.cpp) — /v1/chat/completions 路由

Agent 路径（streaming + non-streaming）：
- 将 `user_prompt` 改为传递完整的最后一条 User `Message`（含 `content_parts`）
- 调用 `agent->invoke(user_message, ...)` 新重载

### 5. [examples/00_smart_app/app_server.cpp](examples/00_smart_app/app_server.cpp) — /v1/memory 路由

读取 memory 后，如果 Message 含 `content_parts`，将 `content` 字段改为 OpenAI 风格数组：
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "这是什么？"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```
如果只含纯文本（`content_parts` 为空），保持原有格式（字符串）。

### 6. [src/memory/memory.h](src/memory/memory.h) — 无变更（接口不变）

IMemory 接口不变，因为 `add(const Message&)` 已经接受 `Message` 对象（含 `content_parts`）。

## 不受影响的文件

- `include/util/common.h` — `Message` 结构已完整定义 `content_parts`
- `include/llm/llm.h` — `ChatRequest` 不变
- LLM 后端 — 各后端已有 `content_parts` 序列化逻辑
- CMakeLists.txt — 无构建变更

## 验证方案

1. **单元测试**: `ctest --test-dir build_x64 -C Debug --output-on-failure` — 已有 7 个测试需全部通过
2. **多模态读写验证**: 手动发送多模态请求（含 `image_url`）到 00_smart_app，然后查询 `/v1/memory` 确认多模态内容被持久化和恢复
3. **SQLite 向后兼容**: 用旧版本生成的 `.db` 文件启动新版，确认能正常读取历史文本对话
4. **Release 编译**: `cmake --build build_x64 --config Release` 确保无 warning/error