# AGENTS.md — Prompt Core

AI agent onboarding. For human onboarding, see [README.md](README.md).

## What This Project Does

This is a library that enables LLM workflow authors to generated structured and validated data via multi-turn LLM conversations, which may be arbitrarily nested, looped, etc in code. A chat step has an LLM guide a user through the process of generating the structured data. The workflow author composes the chat steps and data definitions/validations into arbitrarily complex workflows. 

## Key Files

| File | What |
|------|------|
| `evaluation_criteria/models.py` | Data models & business rules (`EvaluationCriteria`, `Criterion`) |
| `prompt_core/conversation_runtime.py` | `@chat`/`@workflow` decorators, `StructuredConversationOrchestrator`, `StreamingDebug` |
| `prompt_core/llm_interaction.py` | `get_client()` — multi-provider LLM client via instructor+litellm |
| `prompt_core/config.py` | Singleton `Config()` — reads `config.json` for provider/model/timeout |
| `prompt_core/exceptions.py` | Custom exception hierarchy |
| `prompt_core/cli.py` | Typer CLI (`converse` command) |
| `evaluation_criteria/flows.py` | Workflow functions: `generate_criteria`, `refine`, `generate_reviewed_criteria` |

## Conventions

- **Business rules live in Pydantic models** (`model_validator`), not prompts.
- **Prompts give behavioral guidance only** — Instructor handles schema formatting.
- **Must communicate rules in the JSON schema, not just the model_validator.** A `model_validator` only fires *after* the LLM returns data — it's enforcement, not communication. The LLM reads the JSON schema (appended by Instructor to the system message) to understand what to produce. So rules must be encoded in ways that propagate to `model_json_schema()`:
  - Use Pydantic field constraints (`min_length` → `minItems`, `ge`/`le` → `minimum`/`maximum`) — these inject directly into the schema.
  - Use `Field(description=...)` with plain-English conditional rules (e.g. `'Required when action is "success". Must be null when action is "continue".'`).
  - Use class docstrings — Pydantic v2 emits them as the model's `"description"` in JSON schema.
  - Use `model_config = dict(json_schema_extra=...)` for non-standard but model-level annotations the LLM should see.
  - Test schemas with `Model.model_json_schema()` and verify each rule is visible in the output.
- **`max_turns`** appears in both the system prompt (f-string) and the code guard.
- **Tests fail (not skip) without API keys** — this exposes missing infrastructure intentionally.
- **Custom exceptions** for all error cases — CLI formats them, orchestrator raises them.

## Commands

```bash
make                # Set up python venv for development
make test          # Unit tests only (no API key needed, ~0.01s)
make test-verbose  # Same with verbose output per test
make evals          # Real-API evals (requires config.json + API key, ~90s)
make evals-verbose  # Same with verbose output
make lint           # black --check + ruff check
```

## Git Workflow

### Quick Reference

```bash
# Check outstanding work on current branch
git status
git log --oneline -3
BRANCH=$(git branch --show-current)
BASE_BRANCH=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')
git log --oneline "origin/$BASE_BRANCH..origin/$BRANCH"
gh pr list --head "$BRANCH" --state open --json number,url --jq '.[] | [.number, .url] | @tsv'

# Before pushing: run ALL tests locally
make test && make evals

# Start new work from fresh branch off main
git checkout main && git pull origin main && git checkout -b <new-branch>
```

### Rules
- Start every new piece of work from a fresh branch off main.
- If outstanding work exists (unmerged PR, unpushed commits), finish that work first.
- Run both `make test` and `make evals` before pushing.
- origin/main is protected — all changes go through PRs.


## Prompt Design Rules

1. NEVER mention Pydantic class names, field types or validation rules in prompts.
2. DO give behavioral examples and conversation strategies.
3. ALWAYS use f-strings for turn limits (`{self.max_turns}`), never hardcode.
4. FOCUS on "what to do", not "what not to do".
5. If the LLM repeatedly fails to satisfy a Pydantic rule, the rule is not visible enough in the JSON schema. Fix the schema (see Conventions), not the prompt.

## Critical Patterns

- All development must be done on a branch. origin/main is protected.
- `ConversationAction` is a Generic BaseModel with `action: Literal["continue", "success", "failure"]` and a `model_validator` for consistency
- `StructuredConversationOrchestrator.process_turn()` checks turn limit, calls LLM, handles action
- Turn limit raises `TurnLimitExceededError`; failure action raises `ConversationFailedError`

## Decorator API: `@chat` vs `@workflow`

The library provides two decorators for authoring workflow functions:

### `@chat` — Auto-orchestrated leaf functions

Use `@chat` on functions that directly interact with the LLM. The function body is a `pass` stub — the decorator handles everything.

```python
from prompt_core import chat

@chat
def my_workflow_step(
    context: str = "",
    max_turns: int = 10,
) -> EvaluationCriteria:
    """System prompt for the LLM goes here. {param} interpolation works."""
    pass
```

**How it works:**
1. Inspects the return type (must be a Pydantic `BaseModel`)
2. Uses the docstring as the system prompt and appends an auto-generated `## Parameters` section (all params with types, descriptions, and runtime values)
3. Wraps the return type in `ConversationAction[ReturnType]` as the LLM response model
4. Creates a `StructuredConversationOrchestrator` with default callbacks
5. Runs the multi-turn conversation via `ConversationTools.chat()`
6. Returns the inner Pydantic object

**Required parameters:** `io` or `tools` — provides the I/O interface for user interaction.

### `@workflow` — Composite functions

Use `@workflow` on functions that compose multiple `@chat` steps. It injects a `ConversationTools` object so the function can pass `tools=tools` to child functions.

```python
from prompt_core import workflow, ConversationTools

@workflow
def composite_step(
    context: str = "",
    max_turns: int = 10,
    tools: ConversationTools,
) -> EvaluationCriteria:
    criteria = generate_criteria(context=context, max_turns=max_turns, tools=tools)
    # ... additional logic ...
    return criteria
```

**How it works:**
1. If `tools` is already provided (e.g., called from another `@workflow`), passes through
2. Otherwise, creates a `ConversationTools` from the `io` and `state` parameters
3. Calls the function body with `tools` injected

### Parameter Descriptions via `Annotated`

Use `typing.Annotated[T, "description"]` to add descriptions that appear in the auto-generated `## Parameters` section of the system prompt:

```python
from typing import Annotated

@chat
def generate_criteria(
    context: Annotated[
        str, "The topic or domain for which to generate evaluation criteria"
    ] = "",
    max_turns: Annotated[
        int, "Maximum number of conversation turns before giving up"
    ] = 10,
) -> EvaluationCriteria:
    ...
```

This produces `## Parameters` entries like:
```
- `context` (str): The topic or domain for which to generate evaluation criteria
  Value: "choosing a birthday gift"
- `max_turns` (int): Maximum number of conversation turns before giving up
  Value: 10
```

### Docstring Interpolation

The docstring supports:
- `{param_name}` — simple parameter substitution
- `{param.method()}` — method calls on parameter values (e.g., `{initial_object.model_dump()}`)

### Generic Refinement with TypeVar

The `refine()` function uses `TypeVar` to work with any Pydantic model:

```python
ModelType = TypeVar("ModelType", bound=BaseModel)

@chat
def refine(
    initial_object: Annotated[ModelType, "The object to review"],
    max_turns: Annotated[int, "Maximum refinement turns"] = 5,
) -> ModelType:
    """System prompt for refinement..."""
    pass
```

The `@chat` decorator resolves the `TypeVar` to the concrete type passed as `initial_object` at runtime.

### ConversationIO Protocol

All `@chat` functions require an I/O adapter implementing:

```python
class ConversationIO(Protocol):
    def echo(self, message: str) -> None: ...   # Display message to user
    def prompt(self, label: str) -> str: ...     # Get input from user
```

The CLI provides `TyperConversationIO` (using `typer.echo`/`typer.prompt`). For tests, use `unittest.mock.Mock()`.

## Debugging LLM Interactions

When evals hang or behave unexpectedly, enable debug tracing with an environment variable:

```bash
PROMPT_CORE_DEBUG=1 make evals
```

Or for a single test:
```bash
PROMPT_CORE_DEBUG=1 .venv/bin/python -m unittest tests.evals.test_real_api.TestRealAPI.test_name -v
```

This streams all LLM requests/responses to stderr with timing:
```
[15:44:16.001] ━━━ LLM REQUEST ━━━
[15:44:16.001] Model: openrouter/google/gemini-2.0-flash-lite-001
[15:44:16.001] [0] system: You are a helpful assistant...
[15:44:16.001] Waiting for response...
[15:44:17.234] ━━━ LLM RESPONSE (1233ms) ━━━
```

## Reference Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — Deeper architecture and file responsibilities
- [QUICKSTART.md](QUICKSTART.md) — 5-minute contributor guide with critical code locations
- [spec.md](spec.md) — Product specification and philosophy
- [docs/TESTING.md](docs/TESTING.md) — Full testing strategy and patterns
