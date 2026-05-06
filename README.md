# Chat Workflow

A Python library for generating structured evaluation criteria through natural conversation with an LLM.

**Human docs are here. AI agents should read [AGENTS.md](AGENTS.md) for onboarding.**

## What Is This?

This is a library that allows workflow authors to easily build LLM assisted workflows in code. Within a workflow, an LLM will act as facilitator for the creation of complex structured and validated data.
There is a sample workflow where a user can create EvaluationCriteria for a decision.

Built with **Pydantic** (data models + business rules), **Instructor** (structured LLM output), and **litellm** (multi-provider LLM support).

## Quick Start

```bash
# 1. Install
git clone <repo> && cd chat-workflow && make

# 2. Configure
# edit config.json to set your provider + model
export OPENROUTER_API_KEY=your-key    # or OPENAI_API_KEY, etc.

# 3. Activate and run
source .venv/bin/activate
chat-workflow --context "evaluating job offers"
```

See [QUICKSTART.md](QUICKSTART.md) for a 5-minute contributor guide.

## Usage

```bash
# Interactive conversation
chat-workflow --context "choosing a laptop"

# With output file
chat-workflow --context "hiring criteria" --output criteria.json

# Custom max turns
chat-workflow --context "gift ideas" --max-turns 5
```

## Python API

```python
from workflows.evaluation_criteria.flows import generate_reviewed_criteria
from chat_workflow import ConversationFlowState

class MyIO:
    def echo(self, message: str) -> None: print(message)
    def prompt(self, label: str) -> str: return input(label + ": ")

criteria = generate_reviewed_criteria(
    context="evaluating coffee makers",
    max_turns=10,
    io=MyIO(),
    state=ConversationFlowState(),
)
```

## Documentation

| For | Read |
|-----|------|
| Human onboarding (this page) | `README.md` |
| AI agent onboarding | `AGENTS.md` |
| 5-minute contributor guide | `QUICKSTART.md` |
| Architecture & key patterns | `ARCHITECTURE.md` |
| Product specification | `spec.md` |
| Testing strategy | `docs/TESTING.md` |

## License

MIT
