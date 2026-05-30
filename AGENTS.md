# AGENTS.md — Chat Workflow

**Start with [README.md](README.md) for project overview and general information.**

This document provides AI agents with specific guidance for working with the Chat Workflow library. It routes you to the appropriate documentation based on your task.

## What This Project Does

Chat Workflow is a Python library that can be used to write workflows. Workflows can generate structured and validated data through multi-turn LLM conversations with users. Workflow authors compose chat steps and data definitions into arbitrarily complex workflows using standard Python control flow.

## Decision Tree: What Do You Want to Do?

### 1. Run an Existing Workflow
**Goal**: Execute a pre-built workflow to generate structured data.

**Read**: [User Guide](docs/user-guide.md)
- How to install and configure the library
- Using the command-line interface
- Running the example evaluation criteria workflow
- Understanding output formats

### 2. Build a New Workflow
**Goal**: Create a new conversation flow for a specific domain.

**Read**: [Workflow Author Guide](docs/workflow-author-guide.md)
- Understanding the `@atomic_workflow` and `@composite_workflow` decorators
- Defining Pydantic models with business rules
- Writing conversation prompts
- Composing multiple steps into complex workflows
- Testing and debugging workflows

### 3. Hack on the Framework
**Goal**: Contribute to the core library or modify its internals.

**Read**: [Contributor Guide](docs/contributor-guide.md)
- Project architecture and key patterns
- Git workflow and development practices
- Running tests (unit tests and real API evals)
- Making changes to conversation orchestration
- Adding new LLM providers

### 4. See the Example Workflow
**Goal**: Understand the evaluation criteria example that ships with the library.

**Read**: [Example: Evaluation Criteria](docs/example-evaluation-criteria.md)
- Complete walkthrough of the sample workflow
- Business rules and validation logic
- Conversation flow design
- Integration patterns

### 5. Write Documentation
**Goal**: Create or update project documentation.

**Read**: [Writing Documentation Guide](docs/writing-documentation.md)
- Context Efficiency principle
- SOLID/DRY principles for documentation
- Documentation structure and conventions

### 6. Migrate from Couch2food
**Goal**: Update existing workflows from earlier patterns to the new chat-workflow-prototype implementation.

**Read**: [couch2food-migration-guide.md](docs/couch2food-migration-guide.md), [code-generation.md](docs/code-generation.md)
- Understanding the changes from the old architecture to the new
- Using the Blob and Validation annotation system
- Working with BlobSyncMixin and LLMValidated mixins
- Regenerating workflow code via LLM generation and verify_code()

## Critical Agent Information

### Key Files for Quick Understanding

| File | What |
|------|------|
| `workflows/evaluation_criteria/models.py` | Data models & business rules (`EvaluationCriteria`, `Criterion`) |
| `chat_workflow/atomic_workflow.py` | `@atomic_workflow`/`@composite_workflow` decorators, `AtomicWorkflow`, `StreamingDebug` |
| `chat_workflow/llm_interaction.py` | `get_client()` — multi-provider LLM client via instructor+litellm |
| `chat_workflow/config.py` | Singleton `Config()` — reads `config.json` for provider/model/timeout |
| `chat_workflow/exceptions.py` | Custom exception hierarchy |
| `chat_workflow_cli/cli.py` | Typer CLI entrypoint with automatic workflow discovery |
| `workflows/evaluation_criteria/flows.py` | Workflow functions: `generate_criteria`, `refine`, `generate_reviewed_criteria` |

### Development Commands

```bash
make                # Set up python venv for development
make test          # Unit tests only (no API key needed, ~0.01s)
make test-verbose  # Same with verbose output per test
make evals          # Real-API evals (requires config.json + API key, ~90s)
make evals-verbose  # Same with verbose output
make lint           # black --check + ruff check
```

## Reference Documentation Table

| Document | Purpose | Best For |
|----------|---------|----------|
| [README.md](README.md) | Project overview and general introduction | All audiences, GitHub front page |
| [User Guide](docs/user-guide.md) | 5-minute setup and first run | New users getting started |
| [Example: Evaluation Criteria](docs/example-evaluation-criteria.md) | Complete sample workflow walkthrough | Learning by example |
| [Workflow Author Guide](docs/workflow-author-guide.md) | Building new conversation flows | Developers creating workflows |
| [Contributor Guide](docs/contributor-guide.md) | Developing the framework itself | Library contributors |
| [Writing Documentation Guide](docs/writing-documentation.md) | Principles for creating and updating docs | Documentation authors |
| [Testing Guide](docs/TESTING.md) | Testing strategy and patterns | Quality assurance |
| [Product Spec](spec.md) | Product requirements and philosophy | Understanding vision |
| [Couch2food Migration Guide](docs/couch2food-migration-guide.md) | Migrating from old patterns to chat-workflow-prototype | Developers updating existing workflows |
| [Code Generation](docs/code-generation.md) | Generating workflow code via LLM + verify_code() | Developers creating new workflows |

## Git Workflow (For Contributors)

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

For debugging LLM interactions, see the [Contributor Guide](docs/contributor-guide.md#debugging-llm-interactions).
