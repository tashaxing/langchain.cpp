# Plan: Implement util::Timer heartbeat in smart_app

## Context

The user wants to:
1. Implement a `timer` in the `util` module of the langchain.cpp framework
2. Apply this timer in `00_smart_app` to periodically print service alive heartbeat logs after the main thread starts and is ready

## Exploration Results

### Existing Timer Implementation
The `util` module **already has a complete `Timer` implementation**:
- **Header**: `include/util/timer.h` — `langchain::util::Timer` class with:
  - `start_once(delay, cb)` — one-shot timer
  - `start_interval(interval, cb, fire_immediately)` — recurring timer
  - `stop()` — idempotent stop, joins worker thread
  - `is_running()` — query state
- **Source**: `src/util/timer.cpp` — Full implementation using `std::thread`, `std::mutex`, `std::condition_variable`
- The timer is **NOT** included in the umbrella header `include/langchain.h`

### smart_app Structure
- **Entry point**: `examples/00_smart_app/main.cpp`
- Server lifecycle: `app.build()` → `app.start()` → main loop (`while (!g_stop)`) → `app.stop()`
- "Ready" point is after `app.start()` and the console banner is printed (line ~133)
- Logging is via `LOG_INFO(...)` macro from `util/logging.h`
- The app already uses `std::atomic<bool> g_stop` for graceful shutdown

## Implementation Plan

### Step 1: Add timer.h to umbrella header
**File**: `include/langchain.h`
- Add `#include "util/timer.h"` alongside other `util/` includes
- This makes `langchain::util::Timer` available to all consumers of the umbrella header

### Step 2: Integrate heartbeat timer in smart_app
**File**: `examples/00_smart_app/main.cpp`

Changes needed:
1. Include `util/timer.h` (or rely on `langchain.h` which `app_config.h` already pulls in)
2. Add a `langchain::util::Timer heartbeat_timer;` in `main()`
3. After `app.start()` and the "ready" banner, start the timer with `start_interval()`:
   - Interval: 30 seconds (reasonable heartbeat cadence)
   - Callback: `LOG_INFO("Heartbeat: 00_smart_app is alive")`
   - `fire_immediately = true` so first heartbeat prints right after ready
4. Before shutdown (`app.stop()`), call `heartbeat_timer.stop()`

### Code Sketch for main.cpp changes

```cpp
// After app.start();
langchain::util::Timer heartbeat_timer;
heartbeat_timer.start_interval(
    std::chrono::seconds(30),
    [] { LOG_INFO("Heartbeat: 00_smart_app is alive"); },
    true);  // fire immediately

// ... main loop ...

// Before app.stop();
heartbeat_timer.stop();
```

## Verification

1. Build the project:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ```
2. Run `00_smart_app` and observe logs — should see heartbeat every 30s
3. Press Ctrl-C — should shut down cleanly without crashes or hangs

## Files to Modify

| File | Change |
|------|--------|
| `include/langchain.h` | Add `#include "util/timer.h"` |
| `examples/00_smart_app/main.cpp` | Add `Timer` instance, start interval after ready, stop before shutdown |
