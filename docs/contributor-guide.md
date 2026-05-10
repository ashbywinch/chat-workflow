# Contributor Guide

This guide is for developers working **on** the chat-workflow library itself. If you're looking to build workflows using the library, see the [workflow author guide](workflow-author-guide.md).

## What This Project Does

Chat Workflow is a Python library that enables LLM workflow authors to generate structured and validated data via multi-turn LLM conversations. Workflow authors compose chat steps and data definitions into arbitrarily complex workflows.

## Key Files

| File | What |
|------|------|
| `chat_workflow/conversation_runtime.py` | `@chat`/`@workflow` decorators, `StructuredConversationOrchestrator`, `StreamingDebug` |
| `chat_workflow/llm_interaction.py` | `get_client()` — multi-provider LLM client via instructor+litellm |
| `chat_workflow/config.py` | Singleton `Config()` — reads `config.json` for provider/model/timeout |
| `chat_workflow/exceptions.py` | Custom exception hierarchy |
| `chat_workflow/cli.py` | CLI with automatic workflow discovery |
| `chat_workflow/session_logging.py` | Conversation session logging |
| `chat_workflow/prompt_builder.py` | Prompt formatting: `_format_docstring()`, `_build_params_section()` |
| `chat_workflow/metadata.py` | Type introspection: `_format_type_name()`, `_get_return_type()`, etc. |
| `chat_workflow/__init__.py` | Public API exports |

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

#### `chat_workflow/conversation_runtime.py` - Conversation Logic
- Core class: `StructuredConversationOrchestrator`
- Manages turn state (`max_turns` configurable)
- Receives system prompt from `@chat` decorator
- Three outcomes: continue/success/failure

#### `llm_interaction.py` - LLM Abstraction
- Unified client for multiple providers via `get_client()`
- Supports: OpenAI, Google, OpenRouter, etc.
- Uses instructor for structured output

#### `chat_workflow/cli.py` - CLI with Auto-Discovery
- Discovers `@workflow` functions in `workflows/` directory
- Converts function parameters to CLI options (excluding `tools`, `io`, `state`, `debug`)
- Uses `__signature__` override with `typing.get_type_hints()` for type resolution
- Handles `from __future__ import annotations` string annotations

#### `config.py` - Configuration Management
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
## Coding Standards

- We use basedpyright to ensure comprehensive typing.
- Remember that the purpose of types is to help express to the reader (human or agent) what the code does. Make sure our typing is expressive. Multiple levels of nested dicts are not expressive.
- Always use the narrowest type that applies.
- If tempted to use "Any" or "object", double check whether a narrower type would be appropriate.
- If tempted to provide several types in a union, it's likely that a better approach would be to standardise on the one most appropriate type. If the current code uses a variety of types, don't automatically assume that this was a good idea.
- If tempted to put "| None" after your type, check that this isn't a cop-out. Are you sure we should really be allowing None?
- If we read in untyped data (for example, json as a string), coerce it to the narrow type as near to the edge as possible (i.e. in a cli or in unit tests). If we write untyped data, de-type it as close to the edge as possible.
- If tempted to #ignore a basedpyright error, think first. Is there a code or architecture smell that we should fix?
- Prefer to fail fast if something is wrong. Don't silence errors, only use defaults where there is actually a good default option, don't have backstops, don't have three places that you look for something "just in case". Decide what should happen and then fail fast if it doesn't happen.
- If you see a circular import, this is a code smell. Fix the smell, don't bodge the import


## Git Workflow

### Quick Reference

Before starting any new work:
```bash
# Check outstanding work on current branch
git status
git log --oneline -3
BRANCH=$(git branch --show-current)
BASE_BRANCH=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')
git log --oneline "origin/$BASE_BRANCH..origin/$BRANCH"
gh pr list --head "$BRANCH" --state open --json number,url --jq '.[] | [.number, .url] | @tsv'

# Start new work from fresh branch off main
git checkout main && git pull origin main && git checkout -b <new-branch>
```

### Rules
- Start every new piece of work from a fresh branch off main
- If outstanding work exists (unmerged PR, unpushed commits), finish that work first
- origin/main is protected — all changes go through PRs

## Conventions

- **Tests fail (not skip) without API keys** — this exposes missing infrastructure intentionally
- **Custom exceptions** for all error cases — CLI formats them, orchestrator raises them

## Critical Patterns

- All development must be done on a branch. origin/main is protected
- `ConversationAction` is a Generic BaseModel with `action: Literal["continue", "success", "failure"]` and a `model_validator` for consistency
- `StructuredConversationOrchestrator.process_turn()` checks turn limit, calls LLM, handles action
- Turn limit raises `TurnLimitExceededError`; failure action raises `ConversationFailedError`

## SOLID/DRY Principles for Code

These principles guide the module structure and naming conventions in the framework.

### Avoid "utils" Modules

Name modules after their domain, not their category. A module called "utils" has no single responsibility. It becomes a grab bag that grows without bound.

- `prompt_builder.py` not `prompt_utils.py`
- `metadata.py` not `utils/introspection.py`

### Prefer Flat Module Structure

Keep modules flat in the `chat_workflow/` directory rather than nesting them in subdirectories. Deep nesting hides information and makes imports harder to follow.

- `metadata.py` not `utils/introspection.py`
- `prompt_builder.py` not `prompt/prompt_builder.py`

### Single Responsibility Principle

Each module should have one reason to change.

- `prompt_builder.py` owns prompt formatting (docstring rendering, parameter section building)
- `metadata.py` owns type introspection (type name formatting, return type resolution, parameter inspection)
- `conversation_runtime.py` owns conversation orchestration (turn management, LLM calling, action handling)
- `llm_interaction.py` owns LLM provider abstraction

If you find yourself adding a function to a module that doesn't match its stated purpose, create a new module.

### DRY: Extract Shared Logic

When the same pattern appears in multiple places, extract it into a dedicated module. The `prompt_builder.py` and `metadata.py` modules were extracted from `conversation_runtime.py` because prompt formatting and type introspection are used by `decorators.py` and are conceptually separate concerns.

### Rich Models Are OK

Pydantic models can carry convenience methods. A model that validates data can also provide methods that operate on that data. Don't extract every method into a service class just for purity. Use judgment.

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

## Testing Strategy

See the full [testing documentation](TESTING.md) for details.

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

## Common Tasks & Where to Look

| Task | Primary File | Key Function/Method |
|------|--------------|---------------------|
| Add business rule | Example workflow models | `model_validator` methods |
| Modify generic prompt | `chat_workflow/prompt_builder.py` | `_format_docstring()`, `_build_params_section()` |
| Add/modify type introspection | `chat_workflow/metadata.py` | `_format_type_name()`, `_get_return_type()` |
| Add test for new feature | `tests/unit/` | Follow existing test patterns |
| Add eval for new feature | `tests/evals/` | Follow existing eval patterns |
| Modify conversation flow | `chat_workflow/conversation_runtime.py` | `StructuredConversationOrchestrator.process_turn()` |
| Add LLM provider | `chat_workflow/llm_interaction.py` | `get_client()` |
| Modify CLI auto-discovery | `chat_workflow/cli.py` | `build_cli_app()`, `discover_workflow_functions()` |

## Quick Start for Common Changes

### Modify Conversation Flow
1. Edit `@chat`-decorated function docstrings in example workflow files
2. Check `StructuredConversationOrchestrator.process_turn()` logic in `conversation_runtime.py`
3. Update `tests/unit/test_orchestrator_logic.py`

### Add LLM Provider
1. Update `chat_workflow/llm_interaction.py` `get_client()`
2. Add provider configuration handling

### Modify CLI Auto-Discovery
1. Check `chat_workflow/cli.py` `build_cli_app()` function
2. The `@workflow` decorator sets `_is_workflow = True` on functions
3. CLI discovers these functions via `discover_workflow_functions()` 
4. Function parameters (excluding `tools`, `io`, `state`, `debug`) become CLI options
5. Function names are converted to kebab-case for command names

## Reference Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Deeper architecture and file responsibilities
- [QUICKSTART.md](../QUICKSTART.md) — 5-minute contributor guide with critical code locations
- [spec.md](../spec.md) — Product specification and philosophy
- [TESTING.md](TESTING.md) — Full testing strategy and patterns
