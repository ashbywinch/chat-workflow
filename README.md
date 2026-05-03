# Prompt Core

A Python library for generating structured evaluation criteria through natural conversation with an LLM.

**Human docs are here. AI agents should read [AGENTS.md](AGENTS.md) for onboarding.**

## What Is This?

You talk to an AI assistant about a decision (e.g. "birthday presents for a 7-year-old"). It asks questions, you answer. After a few turns it produces a validated set of weighted `EvaluationCriteria` — ready to use for scoring options.

Built with **Pydantic** (data models + business rules), **Instructor** (structured LLM output), and **litellm** (multi-provider LLM support).

## Quick Start

```bash
# 1. Install
git clone <repo> && cd prompt-core && make

# 2. Configure
# edit config.json to set your provider + model
export OPENROUTER_API_KEY=your-key    # or OPENAI_API_KEY, etc.

# 3. Activate and run
source .venv/bin/activate
prompt-core --context "evaluating job offers"
```

See [QUICKSTART.md](QUICKSTART.md) for a 5-minute contributor guide.

## Usage

```bash
# Interactive conversation
prompt-core --context "choosing a laptop"

# With output file
prompt-core --context "hiring criteria" --output criteria.json

# Custom max turns
prompt-core --context "gift ideas" --max-turns 5
```

## Python API

```python
from evaluation_criteria.flows import run_reviewed_criteria_conversation
from prompt_core import ConversationFlowState

class MyIO:
    def echo(self, message: str) -> None: print(message)
    def prompt(self, label: str) -> str: return input(label + ": ")

criteria = run_reviewed_criteria_conversation(
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