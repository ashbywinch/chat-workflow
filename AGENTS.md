# AGENTS.md — Prompt Core

AI agent onboarding. For human onboarding, see [README.md](README.md).

## What This Project Does

Generates structured `EvaluationCriteria` objects via multi-turn LLM conversation. The LLM guides a user through questions, then produces a validated Pydantic model with weighted criteria (min 2, must include "budget").

## Key Files

| File | What |
|------|------|
| `prompt_core/models.py` | Data models & business rules (`EvaluationCriteria`, `Criterion`) |
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

**After making a PR or at the start of a new session**

1. Check for outstanding work on the current branch
Outstanding work may include: uncommitted changes, unpushed commits, pushed commits that are not in a PR yet, or a PR that hasn't merged (perhaps due to failing CI).

```bash
# Check for uncommitted changes
git status

# Check current branch and recent commits
BRANCH=$(git branch --show-current)
git log --oneline -3

# Get default base branch for this repo
BASE_BRANCH=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')

# Check whether this branch exists on origin and has pushed commits not yet in base
git fetch origin
git ls-remote --exit-code --heads origin "$BRANCH"
git log --oneline "origin/$BASE_BRANCH..origin/$BRANCH"

# Check whether this branch has an open PR
gh pr list --head "$BRANCH" --state open --json number,state,url --jq '.[] | [.number, .state, .url] | @tsv'

# If the git log above shows commits but gh pr list prints nothing,
# changes were pushed to the branch but no PR exists yet.

# If a PR exists for this branch, watch its CI checks
gh pr checks --watch
```

2. If there is outstanding work, encourage the user to complete that work first before continuing. Reuse step 1 whenever necessary to ensure that work is completed and fully merged.

3. Only once the existing work is fully merged to main, start the new work:

```bash
# Start new work (only after previous PR is merged)
git branch -d <last-branch>                 # Safe: -d refuses if unmerged
git checkout main && git pull origin main && git checkout -b <new-branch>
```
If branch -d fails because there are outstanding changes then return to step 2

**To push changes**

1. Run ALL tests locally before pushing:
   ```bash
   make test    # Unit tests, fast
   make evals   # Real API tests, requires API key, ~90s
   ```

2. If tests/evals fail locally: fix them before pushing. CI will run them again and fail the same way.

3. Stage, commit, push, and then:
   ```bash
   gh pr checks <number> --watch
   ```

**Rules:**
- Start every new piece of work from a fresh branch off main. Never reuse a branch whose PR has been merged.
- If your PR is open but not yet merged: wait, or ask. Don't push more commits without confirmation.


## Prompt Design Rules

1. NEVER mention Pydantic class names, field types or validation rules in prompts.
2. DO give behavioral examples and conversation strategies.
3. ALWAYS use f-strings for turn limits (`{self.max_turns}`), never hardcode.
4. FOCUS on "what to do", not "what not to do".
5. If the LLM repeatedly fails to satisfy a Pydantic rule, the rule is not visible enough in the JSON schema. Fix the schema (see Conventions), not the prompt.

## Critical Patterns

- All development must be done on a branch. origin/main is protected.
- `ConversationAction` is a single discriminator model with `action: Literal["continue", "success", "failure"]`
- `ConversationOrchestrator.process_turn()` checks turn limit, calls LLM, handles action
- Turn limit raises `TurnLimitExceededError`; failure action raises `ConversationFailedError`

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
