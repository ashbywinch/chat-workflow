# Chat Workflow - Architecture Overview

This document is for developers contributing to the chat-workflow library. It describes the project's architecture, key files, and critical patterns to answer the question: "How is this library structured and how do the pieces fit together?"

## The model layer (read this first)

There is **one** path from the library to a model: `Conversation.send()` in
`chat_workflow/conversation.py`. A workflow step, the rule judge, and an eval's
user simulation are all conversations. Nothing else imports a provider SDK or
calls a model.

```
workflows/, CLI  →  AtomicWorkflow  →  Conversation  →  PydanticAI  →  provider
                                          │
                                          └── transcript (PydanticAI messages)
```

- **Provider mapping** (`model_provider.py`): config → a PydanticAI model.
  OpenAI-compatible services (the house gateway, OpenAI, OpenRouter, Groq,
  Together) and Azure are wired in. Google and Anthropic need their own SDK
  extra and are refused with a message naming it.
- **Schema enforcement** (`conversation.py`): the answer is a call to an
  `answer` tool (`ToolOutput(..., name="answer", strict=True)`), so the request
  carries the schema as that tool's parameters. `response_format` is not used:
  the endpoints behind this model accept it and ignore it (measured — a bare
  object where the schema required the `response` envelope). A tool's schema is
  enforced on the same route. The prompt never restates the schema.
- **A step's tools** (`conversation.py`, `atomic_workflow_config.py`): plain
  functions passed as `tools=[...]`; the signature is the schema the model
  is given, and `deps` is what a tool receives from the caller. A tool that
  raises becomes a `ToolFailed` result the model reads and adapts to; a
  malformed call is a retried tool return under the answer's budget.
- **The transcript is the message list**, not a rendering: thinking, tool
  calls, tool returns, retry prompts and the final output all live in it.
  `transcript.py` is the one projection into text (audit view, user view) and
  the one serialiser (`ModelMessagesTypeAdapter`). `SessionLog.messages` is
  that list.

### Retry policy: one class per failure

| Failure | Response | Budget | Visible |
|---|---|---|---|
| transport (429/5xx/connection/timeout) | the provider SDK resends the identical request | `transport_retries` | no |
| no usable content (blank, thinking-only, token-limit stop) | `RetryModelRequest`: same step, no prompt added, model never learns | `unusable_resends` (per request step) | engine diagnosis only |
| answer rejected (shape or business rule) | a retried `answer` tool return carrying the validator's complaint | `output_retries`, shared with the step's tools | yes (`ToolReturnPart(outcome="retried")`) |
| tool failure | a failed tool *result*; the model adapts in the same run | none | yes (`ToolReturnPart`) |
| malformed tool call | a retried tool return | `output_retries` (one budget, no separate `tool_retries`) | yes |

Shape and business-rule rejections share one output-retry budget because they
are one channel in PydanticAI's native-output path. Separating them would need
a second model loop beside the conversation — the parallel path this design
removes. The two causes stay distinguishable: the retry prompt carries the
validator's own sentence, and the transcript records it.

`max_requests` bounds one turn's model steps (tool rounds included) so a model
that never concludes fails instead of billing forever. A caller that treats
model output as **content under test** — an eval of the model's judgement —
sets `output_retries=0`: a malformed answer is a failure, not a draft.

### Layer boundaries

- `chat_workflow/` (the core) may not import `workflows/` or `chat_workflow_cli/`.
- `workflows/` may import the core; it may not import the CLI.
- `chat_workflow_cli/` is the composition root and may import both.
- The core never calls a model outside `conversation.py`; a new provider is a
  change to `model_provider.py`, not to a workflow.

### **Core Files (Read These First)**
1. `workflows/evaluation_criteria/models.py` - Data models & business rule validation
2. `chat_workflow/atomic_workflow.py` - Conversation orchestration & turn rules
3. `chat_workflow/conversation.py` - The one model call, transcript and retries
4. `chat_workflow/model_provider.py` - Provider/model/endpoint → model
5. `chat_workflow/exceptions.py` - Custom exception hierarchy

### **Configuration Files**
- `config.json` - LLM provider and model settings (REQUIRED)
- Environment variables - API keys (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
- `config.py` - Loads and manages configuration from `config.json`

## Critical Patterns

### 1. Prompt Design Philosophy
Prompts focus on behavioural guidance (role, conversation strategy, interaction
style) rather than data format. The request's JSON schema carries the shape and
the field descriptions; the prompt must not repeat them.

### 2. Validation Layers
```
User Input → Model request (json_schema) → Pydantic validation → typed result
```

### 3. Failure Modes
- **LLM-initiated**: Returns `intent: AgentIntent.FAILURE` for unconstructive users
- **System-initiated**: `TurnLimitExceededError` when `max_turns` exceeded
- **Budget exhausted**: `MaxRetriesExceededError` when the output channel's
  retries run out; transport failures surface as the provider's own error
- **Validation failure**: `ValidationError` for business rules violated outside
  a model turn

For a quick reference of common tasks and where to look, see the [Contributor Guide](docs/contributor-guide.md#common-tasks--where-to-look).
