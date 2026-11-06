# Chat Workflow

> **v0.1 — under active construction.** APIs are unstable, documentation may lag, and some features are incomplete. Contributions and issue reports welcome.

Chat Workflow is a framework for developers to create guided conversation flows, where LLMs act as facilitators for generating complex, structured data. Workflow authors define Pydantic models for their data and write conversation functions that can be combined programmatically to guide users through natural dialogue to produce validated outputs.

## Key Features

- **Natural Conversation Interface**: Users interact through chat-based dialogue, just like talking to a human expert
- **Structured Output Generation**: Every flow produces valid, complete Pydantic objects or raises clear exceptions
- **Validation-First Design**: Information about "What good output looks like" lives in Pydantic models and is validated programmatically
- **Multi-Turn Conversation Management**: Stateful orchestrator handles dialogue flow with configurable turn limits
- **Multi-Provider LLM Support**: Works with OpenAI, Google, OpenRouter, and other providers via litellm
- **Human-in-the-Loop Architecture**: LLMs act as coaches/facilitators, leveraging human knowledge and intuition
- **Deterministic Control Flow**: Authors compose flows using standard Python control structures

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo> && cd chat-workflow

# Set up the development environment
make
```

### Configuration

1. **`config.json` selects the LLM preset.** The default is `cloudflare`
   (Cloudflare AI Gateway → `dynamic/fallback2`, OpenCode primary with
   DeepSeek failover):

```json
{
  "llm": {
    "active": "cloudflare",
    "presets": {
      "cloudflare": {
        "provider": "openai",
        "model": "openai/dynamic/fallback2",
        "api_key_env": "OPENAI_API_KEY"
      }
    }
  }
}
```

2. **Set the gateway credentials** (see `.env.example`; `make setup`
   creates `.env`):

```bash
export OPENAI_BASE_URL=https://gateway.ai.cloudflare.com/v1/<account>/<gateway>/compat
export OPENAI_API_KEY=<gateway token>
```

Switching to a direct provider is a preset change in `config.json` (for
example `openrouter`) plus its key (`OPENROUTER_API_KEY`).

### Run the Example Workflow

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the evaluation criteria workflow
chat-workflow evaluation-criteria generate-reviewed-criteria --context "evaluating job offers"
```

### Command Line Usage

```bash
# List available workflows
chat-workflow --help

# Run the evaluation criteria workflow
chat-workflow evaluation-criteria generate-reviewed-criteria --context "choosing a laptop"

# Customize conversation length
chat-workflow evaluation-criteria generate-reviewed-criteria --context "gift ideas" --max-turns 5
```

## Documentation

| For | Read |
|-----|------|
| **Getting Started** | [User Guide](docs/users/user-guide.md) |
| **Using Existing Workflows** | [Example: Evaluation Criteria](docs/users/example-evaluation-criteria.md) |
| **Building New Workflows** | [Workflow Author Guide](docs/workflow-authors/workflow-author-guide.md) |
| **Contributing to the Framework** | [Contributor Guide](docs/contributors/contributor-guide.md) |
| **Testing Strategy** | [Testing Guide](docs/contributors/testing.md) |
| **Coding Standards** | [Coding Standards](docs/coding-standards.md) |
| **Testing Standards** | [Testing Standards](docs/testing-standards.md) |
| **Product & Architecture** | [PRD](docs/PRD.md) · [TECHSPEC](docs/TECHSPEC.md) · [PLAN](docs/PLAN.md) · [UX](docs/UX.md) |

**AI Agents**: Read [AGENTS.md](AGENTS.md) for agent-specific onboarding and routing.

## License

MIT License
