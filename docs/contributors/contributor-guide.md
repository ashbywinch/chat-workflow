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
| `workflows/workflow/component.py` | `Component` — orchestrator with 4-phase `create()` |
| `workflows/workflow/component_responsibilities.py` | `ComponentResponsibilities` — interface from Workflow to Component |
| `workflows/workflow/models/generated_component.py` | `GeneratedComponent` — stateless code executor |
| `workflows/workflow/models/component_domain_spec.py` | `ComponentDomainSpec` — Phase 1 output (domain understanding) |
| `workflows/workflow/models/component_structure.py` | `ComponentStructure` — Phase 2 output (Pydantic structure) |
| `workflows/workflow/models/component_interaction_context.py` | `ComponentInteractionContext` — Phase 3 output (interaction preferences) |
| `workflows/workflow/models/component_design_spec.py` | `ComponentDesignSpec` — composite of all 3 phases, input to Phase 4 |

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

### Three-Layer Architecture

The framework uses a three-layer architecture for building components, each with distinct responsibilities:

1. **Workflow** (architect) — `Workflow.create()` defines component boundaries and interfaces. It captures what components exist and how they connect, producing `ComponentResponsibilities` for each one. The architecture is considered final before Component starts.

2. **Component** (designer) — `Component.create()` takes the architecture from Workflow and designs component internals through a multi-phase conversation with the user. It produces a complete design spec.

3. **GeneratedComponent** (executor) — `GeneratedComponent.generate()` takes a complete design spec and translates it into Python code statelessly. No design decisions remain. No user conversation about code.

#### Multi-Phase Design

`Component.create()` orchestrates four phases, each a separate `@atomic_workflow` with its own return type and validation rules:

- **Phase 1: Domain Exploration** (`ComponentDomainSpec.explore()`) — Understands what the artifact IS and what makes it good in the user's domain. Returns a `ComponentDomainSpec`.
- **Phase 2: Structural Design** (`ComponentStructure.design()`) — Translates domain concepts into Pydantic field definitions and `@model_validator` rules. Returns a `ComponentStructure`.
- **Phase 3: Interaction Preferences** (`ComponentInteractionContext.gather()`) — Explores how the user wants the assistant to behave during creation. Returns a `ComponentInteractionContext`.
- **Phase 4: Code Generation** (`GeneratedComponent.generate()`) — Statelessly translates the complete design spec into Python code. Returns a `GeneratedComponent`.

Each phase has its own return type with its own validation rules. This keeps the cognitive load on the LLM manageable per step and makes each phase independently testable.

#### Architecture Principles

The architecture follows these principles. Because we are building a **system that builds components**, every rule applies twice: our own components must follow them, AND the validation rules we write must ensure the components we generate also follow them.

1. **Every component has a clear domain purpose and boundary** — Named after an artifact (noun), not a process (verb). Boundary is explicit. Other components interact through its interface, not its internals.

2. **A good workflow has loosely coupled components with clear interfaces** — Minimize dependencies. Each component's interface (what it expects and produces) is explicit and stable. Changes to internals should not ripple to other components.

3. **All conversations with the user stay in the user's domain** — Workflow discusses the user's process, Component discusses the user's artifact. Never about Python, Pydantic, or code structure. Generated components' ``@atomic_workflow`` docstrings must frame the task around the user's domain.

4. **"What Good Looks Like" becomes validation rules** — Proactively ask the user what makes excellent output. Encode quality criteria as ``Field(constraints=...)`` and ``@model_validator`` rules, never as prompt text.

5. **Validation rules never go in prompts** — Business rules → ``Field(constraints=...)`` or ``@model_validator``. "Always ask who attended" → ``attendees: list[str] = Field(min_length=1)``. "Don't hallucinate" → framework-level, not per-component.

6. **Pydantic is the Gatekeeper** — Catch bad LLM output at construction time. Never write fallback logic, workarounds, or fuzzy matching to handle incorrect output. The correct response to bad LLM output is always: improve the prompt.

**Information flow** — ``Workflow.create()`` produces ``ComponentResponsibilities`` (name, purpose, scope, inputs, type). ``Component.create()`` runs multi-phase design: ``DomainSpec.explore()`` → ``Structure.design()`` → ``InteractionContext.gather()`` → ``GeneratedComponent.generate()``. Each phase has its own return type and validation. The generated component is a Pydantic BaseModel with its own ``@atomic_workflow`` method for creating business artifacts.

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
- [testing.md](testing.md) — Full testing strategy and patterns
