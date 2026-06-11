---
name: agent-harness-reflection-loop
description: Implement a 4-phase harness (self-check, feedback, correction, evolution) that wraps any agent runner and uses an evaluator LLM to iteratively improve output quality.
---

# Agent Harness (Self-Check / Self-Feedback / Self-Correction / Self-Evolution)

## Context
User wants to build a "harness" system within the langchain.cpp framework where an agent, after running, can:
1. **Self-check** — evaluate its own output quality
2. **Self-feedback** — generate feedback on what went wrong / could improve
3. **Self-correct** — re-run with adjusted prompts/strategies
4. **Self-evolve** — persist learnings (prompt refinements, strategy adjustments) across runs

This is analogous to:
- Reflection pattern (LLM judges its own output)
- Voyager-style skill learning (evolve tools over time)
- DSPy-style optimization (iteratively improve prompts)

## Current Framework Capabilities
- `ReActAgent` / `ToolCallingAgent` — loop-based agents with tool calling
- `HookManager` — lifecycle hooks (Before/After LLM/Agent/Skill/Tool)
- `IMemory` — conversation memory (BufferMemory, WindowMemory, LongTermMemory)
- `SkillRegistry` / `ChainSkill` — composable skill pipelines
- `ToolRegistry` — tool registration and invocation
- No existing harness / reflection / evolution system

## Design Options

### Option A: Harness as Hook (Observer Pattern)
Implement harness as a set of hooks that observe agent execution, then trigger follow-up LLM calls for evaluation/correction.

**Pros:** Non-invasive, works with any agent, leverages existing hook infrastructure
**Cons:** Limited control over agent internals, harder to implement evolution

### Option B: Harness as Agent Wrapper
Wrap existing agents in a `HarnessAgent` that runs the agent, then runs evaluation/correction loops.

**Pros:** Full control over execution flow, can modify prompts between iterations, easy to implement evolution
**Cons:** More invasive, duplicates some agent logic

### Option C: Harness as Skill (Modular Components)
Build harness components as skills (EvaluatorSkill, CorrectorSkill, EvolverSkill) that can be composed into chains.

**Pros:** Highly modular, reusable, fits existing skill framework
**Cons:** More complex to wire together, may need a meta-controller

## Recommended Approach: Option B (Agent Wrapper) + Option C (Skills for internals)

Create a `HarnessAgent` class that wraps any `AgentRunner` (function or agent) and implements the 4-phase harness loop:

```
Run Phase:
  1. Execute agent → get result + trace
  2. Self-Check: LLM evaluates result against criteria → score + critique
  3. If score < threshold:
     a. Self-Feedback: Generate specific improvement suggestions
     b. Self-Correct: Adjust context/prompts and re-run (with iteration limit)
  4. Self-Evolve: Persist learnings (critique → memory, prompt deltas → registry)
```

### Components to Add

1. **`include/harness/harness.h`** — Core harness types and interfaces
   - `struct HarnessConfig` — thresholds, max correction iterations, evolution settings
   - `struct CheckResult` — score (0-1), critique, suggestions
   - `class HarnessAgent` — wraps an agent with the 4-phase loop
   - `class IEvaluator` — interface for evaluation strategies
   - `class LLMEvaluator` — default: uses LLM to evaluate output
   - `class IPromptEvolver` — interface for prompt evolution
   - `class ReflectionEvolver` — evolves prompts based on critique history

2. **`src/harness/harness.cpp`** — Implementation

3. **`include/harness/reflection.h`** — Reflection-specific utilities
   - Prompt templates for self-check, self-feedback
   - Trace formatter (converts AgentResult to evaluable text)

4. **`examples/13_harness_demo.cpp`** — Demonstration

### Integration Points

- Uses existing `HookManager` to capture execution traces
- Uses existing `IMemory` to persist critique history
- Uses existing `LLM` for evaluation/correction calls
- Uses existing `SkillRegistry` to store evolved prompts as skills

### File Changes

**New files:**
- `include/harness/harness.h`
- `include/harness/reflection.h`
- `src/harness/harness.cpp`
- `examples/13_harness_demo.cpp`

**Modified files:**
- `include/langchain.h` — add harness headers
- `CMakeLists.txt` — add harness source (GLOB handles this automatically)

### Verification

1. Build: `cmake --build build --config Debug`
2. Run demo: `./build/Debug/13_harness_demo.exe`
3. Verify: Agent runs, evaluates itself, corrects if needed, persists learnings
