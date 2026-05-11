# Contributor Guide

This guide is for developers working **on** the chat-workflow library itself. If you're looking to build workflows using the library, see the [workflow author guide](workflow-author-guide.md).

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
| `chat_workflow/cli.py` | CLI with automatic workflow discovery |
| `chat_workflow/session_logging.py` | Session logging to disk |
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

#### `chat_workflow/atomic_workflow.py` - Conversation Logic
- Core class: `AtomicWorkflow`
- Manages turn state (`max_turns` configurable)
- Receives system prompt from `@atomic_workflow` decorator
- Three outcomes: continue/success/failure

#### `llm_interaction.py` - LLM Abstraction
- Unified client for multiple providers via `get_client()`
- Supports: OpenAI, Google, OpenRouter, etc.
- Uses instructor for structured output

#### `chat_workflow/cli.py` - CLI with Auto-Discovery
- Discovers `@composite_workflow` functions in `workflows/` directory
- Converts function parameters to CLI options (excluding `session`, `io`, `state`, `debug`)
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

### Types
- We use basedpyright to ensure comprehensive typing.
- Remember that the purpose of types is to help express to the reader (human or agent) what the code does. Make sure our typing is expressive. Multiple levels of nested dicts are not expressive.
- Always use the narrowest type that applies.
- If tempted to use "Any" or "object", double check whether a narrower type would be appropriate.
- If tempted to provide several types in a union, it's likely that a better approach would be to standardise on the one most appropriate type. If the current code uses a variety of types, don't automatically assume that this was a good idea.
- If tempted to put "| None" after your type, check that this isn't a cop-out. Are you sure we should really be allowing None?
- If we read in untyped data (for example, json as a string), coerce it to the narrow type as near to the edge as possible (i.e. in a cli or in unit tests). If we write untyped data, de-type it as close to the edge as possible.
- If tempted to #ignore a basedpyright error, think first. Is there a code or architecture smell that we should fix?

### Principles
- Naming is very important. Each class, function and variable should be named carefully in order to help readers understand the structure of the code as a whole.
- Each class should be on its own in a module named after that class
- Functionality that is tightly coupled to the contents of a class should be in a member function of that class
- Code should be written in a functional programming style wherever reasonably possible
- **Prefer libraries over reinvention**: Before writing non-trivial code from scratch, check whether a library already solves the problem. Adding a dev dependency has no user-facing cost. Adding a production dependency is often the right call too. The decision criterion is simplicity and readability: a library call that replaces 30 lines of custom code is worth it; a library that adds more complexity than the code it replaces is not.
- Prefer to fail fast if something is wrong. Don't silence errors, only use defaults where there is actually a good default option, don't have backstops, don't have three places that you look for something "just in case". Decide what should happen and then fail fast if it doesn't happen.
- We do not maintain backwards compatability with previous versions of anything
- Module and package exports should be organised so that the public API surface is importable from the package root. If code is moved to a different submodule, only ``__init__.py`` should need to change. External consumers must import from the package root (``from mypackage import Thing``), not from submodules (``from mypackage.submodule import Thing``). Internal code within the package should use relative submodule imports as normal.
- If a class or function name uses vague terms like "Manager", "Enhanced" or "Configured", reconsider whether the base concept is well-defined. ``AtomicWorkflow`` without "Structured" says everything ``StructuredConversationOrchestrator`` said. When two concepts genuinely need disambiguation, the names should complement each other (e.g., ``AtomicWorkflow`` and ``CompositeWorkflow`` — each clarifies the other).
- If the best docstring you can write just rephrases the name (``"""ConversationOrchestrator orchestrates conversations."""``), that is a smell. Either the name is too vague or the concept boundaries are unclear. 

### Smells
- If you see a circular import, this is a code smell. Fix the smell, don't bodge the import.
- A long file is a strong signal that multiple concerns have become mixed together. Identify subsets of the code that will change for different reasons and move each axis of change into its own module. Type-resolution logic changes when you add new type patterns; decorator logic changes when you alter the flow. Those belong in different files regardless of line count.
- A docstring that adds no information beyond the name is a smell. The class may need a better name, clearer boundaries, or both. (Sometimes the class is really self describing with no need for a docstring, and that's great!)


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

### Updating a PR Description

The `gh pr edit --body` flag can silently fail (e.g., when the remote URL is
stale after a repo rename). To reliably update a PR's body, write it to a file
and use the API directly instead:

```bash
# Write the new body to a file
cat > /tmp/pr_body.md << 'EOF'
## Summary
...
EOF

# Update body
gh api "repos/$(gh repo view --json owner,name --jq '[.owner.login,.name] | join("/")')/pulls/$PR_NUMBER" \
  -X PATCH -F body=@/tmp/pr_body.md

# Update title (works with both methods)
gh api "repos/$(gh repo view --json owner,name --jq '[.owner.login,.name] | join("/")')/pulls/$PR_NUMBER" \
  -X PATCH -f title="New title here"
```

The `-F body=@file` form reliably sends the file contents as a string field. The
`-f` flag is for short string fields. Use `-F` (capital) for file references with
`@` and `-f` for inline values.

## Conventions

- **Tests fail (not skip) without API keys** — this exposes missing infrastructure intentionally
- **Custom exceptions** for all error cases — CLI formats them, orchestrator raises them

## Critical Patterns

- All development must be done on a branch. origin/main is protected
- `AgentResponse` is a Generic BaseModel with `intent: AgentIntent` and a `model_validator` for consistency
- `AtomicWorkflow.process_turn()` checks turn limit, calls LLM, handles intent
- Turn limit raises `TurnLimitExceededError`; failure intent raises `AtomicWorkflowFailedError`

## SOLID/DRY Principles for Code

These principles guide the module structure and naming conventions in the framework.

### Module Naming

Avoid generic words like "utils", "manager", "tools" in module names. Use domain-driven names instead.

- `prompt_builder.py` not `prompt_utils.py`
- `metadata.py` not `utils/introspection.py`

A module named "utils" is a grab bag. It has no single responsibility. It grows without bound. Name modules after what they do.

### Prefer Flat Module Structure

Keep modules flat in the `chat_workflow/` directory rather than nesting them in subdirectories. Deep nesting hides information and makes imports harder to follow.

- `metadata.py` not `utils/introspection.py`
- `prompt_builder.py` not `prompt/prompt_builder.py`

### Single Responsibility Principle

Each module should have one reason to change.

- `prompt_builder.py` owns prompt formatting (docstring rendering, parameter section building)
- `metadata.py` owns type introspection (type name formatting, return type resolution, parameter inspection)
- `atomic_workflow.py` owns conversation orchestration (turn management, LLM calling, intent handling)
- `llm_interaction.py` owns LLM provider abstraction

If you find yourself adding a function to a module that doesn't match its stated purpose, create a new module.

### DRY: Extract Shared Logic

When the same pattern appears in multiple places, extract it into a dedicated module. The `prompt_builder.py` and `metadata.py` modules were extracted from `atomic_workflow.py` because prompt formatting and type introspection are used by `decorators.py` and are conceptually separate concerns.

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
| Modify conversation flow | `chat_workflow/atomic_workflow.py` | `AtomicWorkflow.process_turn()` |
| Add LLM provider | `chat_workflow/llm_interaction.py` | `get_client()` |
| Modify CLI auto-discovery | `chat_workflow/cli.py` | `build_cli_app()`, `discover_workflow_functions()` |

## Quick Start for Common Changes

### Modify Conversation Flow
1. Edit `@atomic_workflow`-decorated function docstrings in example workflow files
2. Check `AtomicWorkflow.process_turn()` logic in `atomic_workflow.py`
3. Update `tests/unit/test_orchestrator_logic.py`

### Add LLM Provider
1. Update `chat_workflow/llm_interaction.py` `get_client()`
2. Add provider configuration handling

### Modify CLI Auto-Discovery
1. Check `chat_workflow/cli.py` `build_cli_app()` function
2. The `@composite_workflow` decorator sets `_is_workflow = True` on functions
3. CLI discovers these functions via `discover_workflow_functions()` 
4. Function parameters (excluding `session`, `io`, `state`, `debug`) become CLI options
5. Function names are converted to kebab-case for command names

## Reference Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Architecture overview, core files, and critical patterns
- [QUICKSTART.md](../QUICKSTART.md) — 5-minute contributor guide with critical code locations
- [spec.md](../spec.md) — Product specification and philosophy
- [TESTING.md](TESTING.md) — Full testing strategy and patterns
