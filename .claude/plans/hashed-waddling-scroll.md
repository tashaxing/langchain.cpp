# Plan: 为 ILLM 增加异步 API（在同步之上扩展，不取代）

## Context

当前 `ILLM` 接口仅有同步的 `invoke()` / `invoke_stream()`。在 `ApiServer` 等服务器场景中，每个 HTTP worker 线程会被 LLM 调用阻塞（网络 I/O + 等待响应），限制了并发处理能力。用户要求**在原有同步 API 之上增加异步 API**，使用时灵活选择，不是取代关系。

## 约束

- **C++17** — 不能用 `std::jthread`, coroutines, `std::format` 等 C++20 特性
- **不引入新依赖** — 只用标准库的 `<future>`, `<thread>`, `<mutex>`, `<condition_variable>`
- **保持 NVI 模式** — 同步的 `invoke()` / `invoke_stream()` 仍由基类统一触发 Hook
- **Hook 语义不变** — Before*/After* 仍应在调用前后正确触发
- **向后兼容** — 所有现有代码不修改即可编译运行
- **Header/Implementation 分离** — 非模板函数定义放在 `.cpp`

## 设计决策

### 方案选择：Callback-based（推荐）

对比 `std::future` 和 callback-based：

| 维度 | std::future | callback-based |
|------|------------|----------------|
| 流式支持 | 困难（future 只能 resolve 一次） | 自然（每来一块数据就回调） |
| 与现有 StreamCallback 对齐 | 不对齐 | 对齐 |
| 内存占用 | 需要存储完整响应 | 流式消费，低内存 |
| 错误处理 | 通过 future::get() 抛异常 | 通过回调参数传递 |
| C++17 兼容性 | ✅ | ✅ |

**结论**：采用 callback-based。提供两个异步方法：
- `async_invoke(req, on_complete)` — 非流式，完成时回调
- `async_invoke_stream(req, on_delta, on_complete)` — 流式，每块数据回调 + 完成时回调

### 接口设计

```cpp
// include/llm/llm.h

// 异步完成回调：成功时 response 非空，失败时 error 非空
using AsyncCompleteCallback = std::function<void(const ChatResponse* response,
                                                  const std::string* error)>;

class ILLM
{
public:
    // ---- 原有同步 API（不变）----
    ChatResponse invoke(const ChatRequest& req);
    ChatResponse invoke_stream(const ChatRequest& req, const StreamCallback& on_delta);
    std::string complete(const std::string& prompt);

    // ---- 新增异步 API ----
    void async_invoke(const ChatRequest& req, const AsyncCompleteCallback& on_complete);
    void async_invoke_stream(const ChatRequest& req,
                              const StreamCallback& on_delta,
                              const AsyncCompleteCallback& on_complete);

    // ... 其余不变

protected:
    // ---- 子类必须实现的同步方法（不变）----
    virtual ChatResponse invoke_impl(const ChatRequest& req) = 0;
    virtual ChatResponse invoke_stream_impl(const ChatRequest& req,
                                            const StreamCallback& on_delta);

    // ---- 子类可选实现的异步方法 ----
    // 默认实现：在后台线程调用同步 invoke_impl / invoke_stream_impl
    // 真正支持异步 I/O 的后端（如基于 libcurl-multi 或 asio）可以覆盖此方法
    virtual void async_invoke_impl(const ChatRequest& req,
                                    const AsyncCompleteCallback& on_complete);
    virtual void async_invoke_stream_impl(const ChatRequest& req,
                                           const StreamCallback& on_delta,
                                           const AsyncCompleteCallback& on_complete);
};
```

### Hook 处理

异步方法的 Hook 语义：
- `BeforeLLM` 在提交异步任务前触发（调用者线程）
- `AfterLLM` 在异步任务完成后触发（后台线程）
- `ScopedSpan` 需要改造以支持异步场景，或手动拆分 Before/After

实现策略：
```cpp
void ILLM::async_invoke(const ChatRequest& req, const AsyncCompleteCallback& on_complete)
{
    ChatRequest local = req;
    auto* mgr = hooks_ ? hooks_ : &hook::HookManager::global();

    // Before 在调用者线程触发
    hook::HookContext before;
    before.phase = hook::Phase::BeforeLLM;
    before.component = name();
    before.call_id = mgr->new_call_id();
    before.mutable_request = &local;
    mgr->fire(before);

    std::string call_id = before.call_id;

    // 包装用户回调，在完成后触发 After
    auto wrapped = [this, mgr, call_id, on_complete](
        const ChatResponse* resp, const std::string* err)
    {
        hook::HookContext after;
        after.phase = hook::Phase::AfterLLM;
        after.component = name();
        after.call_id = call_id;
        after.response = resp;
        if (err && !err->empty())
        {
            after.metadata["error"] = *err;
        }
        mgr->fire(after);

        if (on_complete)
        {
            on_complete(resp, err);
        }
    };

    async_invoke_impl(local, wrapped);
}
```

### 默认 async_impl：线程池后台执行

```cpp
// src/llm/llm.cpp

void ILLM::async_invoke_impl(const ChatRequest& req,
                              const AsyncCompleteCallback& on_complete)
{
    // 默认实现：在独立线程中执行同步 invoke_impl
    std::thread([this, req, on_complete]() mutable
    {
        try
        {
            auto resp = invoke_impl(req);
            if (on_complete)
            {
                on_complete(&resp, nullptr);
            }
        }
        catch (const std::exception& e)
        {
            if (on_complete)
            {
                std::string err = e.what();
                on_complete(nullptr, &err);
            }
        }
    }).detach();
}
```

**注意**：频繁创建/销毁线程成本高。考虑在 `util/` 中增加一个轻量级线程池（`ThreadPool`），供异步默认实现使用。

### 线程池设计（轻量级）

```cpp
// include/util/thread_pool.h
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace langchain
{
namespace util
{

class ThreadPool
{
public:
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task);

    static ThreadPool& default_pool(); // 进程级默认线程池

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace util
} // namespace langchain
```

## 修改文件清单

### 新增文件
1. `include/util/thread_pool.h` — 轻量级线程池
2. `src/util/thread_pool.cpp` — 线程池实现

### 修改文件
3. `include/llm/llm.h` — 增加 `AsyncCompleteCallback` 和 `async_invoke` / `async_invoke_stream`
4. `src/llm/llm.cpp` — 实现异步方法的 NVI 层 + 默认 async_impl
5. `include/api/server.h` — 无需修改（但后续可利用异步 API）
6. `src/api/server.cpp` — 后续可改造 `/v1/chat/completions` 使用 `async_invoke`

### 测试
7. `tests/test_async_llm.cpp` — 新增测试：
   - 默认 async_impl 是否正确在后台线程执行
   - Hook Before/After 是否跨线程正确触发
   - 异常是否通过 error 回调正确传递
   - 流式异步是否正确

## 验证计划

```bash
# 1. 编译
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. 运行测试
ctest --test-dir build -R test_async_llm --output-on-failure

# 3. 运行原有测试确保无回归
ctest --test-dir build --output-on-failure
```

## 后续扩展（不在本次计划内）

- `ApiServer` 的 `/v1/chat/completions` 可改造为使用 `async_invoke`，释放 HTTP worker 线程
- 各 LLM 后端（OpenAILLM 等）可覆盖 `async_invoke_impl`，使用 `libcurl-multi` 实现真正的异步 I/O
- Agent 层可增加 `async_invoke` 方法，实现并发的 tool calling
