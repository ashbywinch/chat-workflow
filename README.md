# Chat Workflow

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

1. **Edit `config.json`** to set your preferred LLM provider and model:

```json
{
  "provider": "openrouter",
  "model": "google/gemini-2.0-flash-lite-001",
  "temperature": 0.7,
  "max_retries": 3
}
```

2. **Set your API key** environment variable:

```bash
export OPENROUTER_API_KEY=your-key-here  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

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

**AI Agents**: Read [AGENTS.md](AGENTS.md) for agent-specific onboarding and routing.

## License

MIT License
