# Domain Concepts

This document defines the core vocabulary of the chat-workflow library and maps it to equivalent concepts in the broader LLM ecosystem. Understanding these terms helps newcomers leverage existing domain knowledge when reading or contributing to the codebase.

## Atomic Workflow

**What it is:** The smallest unit of LLM-driven conversation that produces a validated Pydantic object. An atomic workflow runs a multi-turn loop (user ↔ LLM) until the LLM signals success with a valid result, failure, or the turn limit is reached.

**Broadly known as:** In other frameworks this concept appears as a *structured output agent loop*, a *function-calling agent*, a *tool-use loop*, or a *run* (OpenAI SDK). Vercel's AI SDK calls it a multi-step *generation* with `stopWhen`. LangGraph implements it as a *StateGraph* that loops until a condition is met. LlamaIndex has *FunctionAgent* for the same pattern.

**In the code:** `AtomicWorkflow` (class), `@atomic_workflow` (decorator), `process_turn()` (method that drives one user → LLM cycle).

## Composite Workflow

**What it is:** A function that composes multiple atomic workflows and other composite workflows into a higher-level process. It does not drive its own LLM loop — it passes a `Session` object to child workflows and orchestrates their execution.

**Broadly known as:** *Multi-agent workflow* (LlamaIndex), *workflow graph* (LangGraph), *pipeline* or *orchestration*. The pattern of composing reusable steps into a larger process is universal in workflow systems (Amazon States Language, Temporal, Airflow).

**In the code:** `@composite_workflow` (decorator), `Session.run()` (drives an atomic workflow within a composite), `generate_reviewed_criteria` (example).

## Turn

**What it is:** A single LLM round trip: the framework sends a user message (plus accumulated history) to the LLM, receives a response, and acts on it. If the response is "continue", another turn begins.

**Broadly known as:** *Turn* is the standard term (OpenAI SDK, Agent Client Protocol, Microsoft Agent Framework). Vercel's AI SDK calls it a *step*. LangGraph calls it a *super-step*.

**In the code:** `turn_count` (field), `TurnLimitExceededError`, `TurnResult`, `process_turn()`. The concept is already correctly named.

## Session

**What it is:** The runtime context passed to every workflow function. A session wraps three things: IO (how to talk to the user), state log (accumulated conversation history across all turns), and config (provider, model, timeout). It persists for the lifetime of a single CLI invocation.

**Broadly known as:** *Session* is widely used (OpenAI SDK, Microsoft Agent Framework, Agent Client Protocol). It represents durable, multi-turn state that survives individual LLM calls. Other frameworks call the same concept *runtime context* or *invocation context*.

**In the code:** `Session` (class), `SessionLog` (the accumulated transcript), injected as the `session` parameter in workflow functions.

## Agent Response

**What it is:** The structured object the LLM returns after one turn. It carries an `intent` (what the framework should do next) and optionally a `message` for the user or a `result` (the validated Pydantic object).

**Broadly known as:** *Structured output* (Vercel AI SDK), *tool call* / *function call* (OpenAI API), *completion* with a *finish reason*. The `intent` field aligns with what other frameworks call *stop reason*, *finish reason*, or *routing decision*.

**In the code:** `AgentResponse` (model), `AgentIntent` (enum: `CONTINUE`, `SUCCESS`, `FAILURE`).

## Turn Result

**What it is:** The framework's internal outcome after processing one turn — wraps the LLM's response into a standard shape (`is_complete`, `message`, `result`) used by the turn loop.

**Broadly known as:** This is plumbing analogous to a *step result* or *node output* in other frameworks. It's the return value of one iteration of the agent loop.

**In the code:** `TurnResult` (model), with factory methods `continuing()`, `success()`, `failure()`.

## How These Fit Together

```
Session (one CLI invocation)
  └── Composite Workflow
        ├── Atomic Workflow          ← has its own turn loop
        │     ├── Turn 1 (user → LLM → AgentResponse)
        │     ├── Turn 2 (user → LLM → AgentResponse)
        │     └── ... → returns TurnResult
        ├── Atomic Workflow
        │     └── ...
        └── (can nest further)
```

The outer container is a **Session** (one user session). Inside it, a **Composite Workflow** coordinates one or more **Atomic Workflows**. Each atomic workflow runs a **turn** loop until the LLM returns an **Agent Response** with `intent=SUCCESS` or `intent=FAILURE`. That response is wrapped in a **Turn Result** and recorded in the session's log.