# Contributor Guide

This guide is for developers working **on** the chat-workflow library itself. If you're looking to build workflows using the library, see the [workflow author guide](../workflow-authors/workflow-author-guide.md).

If you are new to the codebase, start with [../reference/domain-concepts.md](../reference/domain-concepts.md) — it defines the core vocabulary (atomic workflow, turn, session, etc.) and maps each term to equivalent concepts in the broader LLM ecosystem.

For coding standards, see [coding-standards.md](coding-standards.md).
For git workflow and PR guidance, see [git-workflow.md](git-workflow.md).

---

## What This Project Does

Chat Workflow is a Python library that enables LLM workflow authors to generate structured and validated data via multi-turn LLM conversations. Workflow authors compose chat steps and data definitions into arbitrarily complex workflows.

## Key Files

| File | What |
|------|------|
| `chat_workflow/decorators.py` | `@atomic_workflow`/`@composite_workflow` decorators |
| `chat_workflow/atomic_workflow.py` | `AtomicWorkflow` — drives a single atomic workflow turn loop |
| `chat_workflow/session.py` | `Session` — runtime context (IO, state, config) |
| `chat_workflow/session_log.py` | `SessionLog` — accumulated session state |
| `chat_workflow/debug.py` | `StreamingDebug` — real-time LLM tracing |
| `chat_workflow/llm_interaction.py` | `get_client()` — multi-provider LLM client via instructor+litellm |
| `chat_workflow/config.py` | Singleton `Config()` — reads `config.json` for provider/model/timeout |
| `chat_workflow/exceptions.py` | Custom exception hierarchy |
| `chat_workflow_cli/cli.py` | CLI with automatic workflow discovery |
| `chat_workflow/session_logging.py` | Session logging to disk |
| `chat_workflow/prompt_builder.py` | Prompt formatting: `_format_docstring()`, `_build_params_section()` |
| `chat_workflow/metadata.py` | Type introspection: `_format_type_name()`, `_get_return_type()`, etc. |
| `chat_workflow/__init__.py` | Public API exports |
| `chat_workflow/annotations.py` | Blob and Validation annotations |
| `chat_workflow/mixins.py` | BlobSyncMixin and LLMValidated mixins |

Example workflows live in the `workflows/` directory.

## Architecture Overview

### Core Concepts

- **High usability for workflow authors**: Authors must be able to easily figure out the library and create extremely readable workflows with minimal boilerplate
- **Structured outputs via Instructor**: LLM returns Pydantic objects
- **Multi-turn conversation**: Stateful orchestrator manages dialogue flow
- **Dual configuration**: Provider/model in `config.json`, API keys in environment
- **Configuration at the edge**: Configuration must only be set/read at the perimeter (CLI, test setup, etc.)
- **Fail fast**: We are never backwards compatible. If something is configured incorrectly we fail fast instead of using defaults
- **User-friendly error messages**: All error messages shown to users or workflow authors should be clear and helpful

### File Responsibilities

#### `chat_workflow/atomic_workflow.py` — Conversation Logic
- Core class: `AtomicWorkflow`
- Manages turn state (`max_turns` configurable)
- Receives system prompt from `@atomic_workflow` decorator
- Three outcomes: continue/success/failure

#### `llm_interaction.py` — LLM Abstraction
- Unified client for multiple providers via `get_client()`
- Supports: OpenAI, Google, OpenRouter, etc.
- Uses instructor for structured output

#### `chat_workflow_cli/cli.py` — CLI with Auto-Discovery
- Discovers `@composite_workflow` functions in `workflows/` directory
- Converts function parameters to CLI options (excluding `session`, `io`, `state`, `debug`)
- Uses `__signature__` override with `typing.get_type_hints()` for type resolution
- Handles `from __future__ import annotations` string annotations

#### `config.py` — Configuration Management
- Singleton configuration manager
- Reads ONLY from `config.json`
- Provides: provider, model, temperature, `max_retries`
- Note: API keys come from environment variables, not config.json

## Development Commands

```bash
make                # Set up python venv for development
make test          # Unit tests only (no API key needed, ~0.01s)
make test-verbose  # Same with verbose output per test
make evals          # Real-API evals (requires config.json + API key, ~90s)
make evals-verbose  # Same with verbose output
make lint           # ruff and basedpyright
```

## Critical Patterns

- **Tests fail (not skip) without API keys** — this exposes missing infrastructure intentionally
- **Custom exceptions** for all error cases — CLI formats them, orchestrator raises them
- `AgentResponse` is a Generic BaseModel with `intent: AgentIntent` and a `model_validator` for consistency
- `AtomicWorkflow.process_turn()` checks turn limit, calls LLM, handles intent
- Turn limit raises `TurnLimitExceededError`; failure intent raises `AtomicWorkflowFailedError`

## Testing Strategy

See the full [testing documentation](testing.md) for details.

### Philosophy
1. **Fail fast with exceptions** — code raises, doesn't return failure objects
2. **Clear separation** — unit tests (mocked) vs evals (real API)
3. **Infrastructure exposure** — evals **fail (not skip)** when API keys missing
4. **Proper mocking** — test logic with mocks, test prompts with real API

### Test Structure
```
tests/
├── unit/
│   ├── test_models.py              # Pydantic model validation (no mock)
│   ├── test_orchestrator_logic.py  # Orchestrator logic (mock _call_llm)
│   └── test_llm_interaction.py     # LLM interaction (mock get_client)
└── evals/
    └── test_real_api.py            # Real API tests (require config.json + API key)
```

### Test Coverage Goals
- **Models**: 100% (validation, business rules)
- **Orchestrator logic**: 100% (turn management, action handling)
- **LLM interaction**: 100% (retry logic, error handling)
- **Integration**: Critical paths only (real API interaction)

## Debugging LLM Interactions

When evals hang or behave unexpectedly, enable debug tracing with an environment variable:

```bash
CHAT_WORKFLOW_DEBUG=1 make evals
```

Or for a single test:
```bash
CHAT_WORKFLOW_DEBUG=1 .venv/bin/python -m unittest tests.evals.test_real_api.TestRealAPI.test_name -v
```

This streams all LLM requests/responses to stderr with timing:
```
[15:44:16.001] ━━━ LLM REQUEST ━━━
[15:44:16.001] Model: openrouter/google/gemini-2.0-flash-lite-001
[15:44:16.001] [0] system: You are a helpful assistant...
[15:44:16.001] Waiting for response...
[15:44:17.234] ━━━ LLM RESPONSE (1233ms) ━━━
```

## Common Tasks & Where to Look

| Task | Primary File | Key Function/Method |
|------|--------------|---------------------|
| Add business rule | Example workflow models | `model_validator` methods |
| Modify generic prompt | `chat_workflow/prompt_builder.py` | `_format_docstring()`, `_build_params_section()` |
| Add/modify type introspection | `chat_workflow/metadata.py` | `_format_type_name()`, `_get_return_type()` |
| Add test for new feature | `tests/unit/` | Follow existing test patterns |
| Add eval for new feature | `tests/evals/` | Follow existing eval patterns |
| Modify conversation flow | `chat_workflow/atomic_workflow.py` | `AtomicWorkflow.process_turn()` |
| Add LLM provider | `chat_workflow/llm_interaction.py` | `get_client()` |
| Modify CLI auto-discovery | `chat_workflow_cli/cli.py` | `build_cli_app()`, `discover_workflow_functions()` |

### Quick Start for Common Changes

**Modify conversation flow:**
1. Edit `@atomic_workflow`-decorated function docstrings in example workflow files
2. Check `AtomicWorkflow.process_turn()` logic in `atomic_workflow.py`
3. Update `tests/unit/test_orchestrator_logic.py`

**Add LLM provider:**
1. Update `chat_workflow/llm_interaction.py` `get_client()`
2. Add provider configuration handling

**Modify CLI auto-discovery:**
1. Check `chat_workflow_cli/cli.py` `build_cli_app()` function
2. The `@composite_workflow` decorator sets `_is_workflow = True` on functions
3. CLI discovers these functions via `discover_workflow_functions()`
4. Function parameters (excluding `session`, `io`, `state`, `debug`) become CLI options
5. Function names are converted to kebab-case for command names

## Reference Docs

- [Domain Concepts](../reference/domain-concepts.md) — Vocabulary defined and mapped to the broader LLM ecosystem
- [Coding Standards](coding-standards.md) — Types, principles, naming, smells
- [Git Workflow](git-workflow.md) — Branching, commits, PR descriptions
- [spec.md](../../spec.md) — Product specification and philosophy
- [testing.md](testing.md) — Full testing strategy and patterns
