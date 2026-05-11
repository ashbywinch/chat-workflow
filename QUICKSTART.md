# Quick Start for New Contributors

This guide is for new contributors who want to start working on the chat-workflow library quickly. It covers the essential files, commands, and code locations to answer the question: "What do I need to know to make my first change?"

## 30-Second Overview
- **Goal**: Generate evaluation criteria via LLM conversation
- **Core**: `AtomicWorkflow` manages multi-turn dialogue  
- **Output**: `EvaluationCriteria` Pydantic model (validated)
- **Key rule**: Must include "budget" criterion, ≥2 total criteria

## First 5 Files to Read
1. `workflows/evaluation_criteria/models.py` - Data structures & validation rules
2. `chat_workflow/atomic_workflow.py` - Conversation orchestration & decorators
3. `tests/unit/test_models.py` - See business rule tests
4. `tests/unit/test_orchestrator_logic.py` - See conversation flow tests
5. `spec.md` - Product requirements

For a deeper understanding, see the [Contributor Guide](docs/contributor-guide.md) and [Architecture Overview](ARCHITECTURE.md).

## Critical Code Locations
```python
# Business rules (MUST maintain):
workflows/evaluation_criteria/models.py:44  # validate_business_rules() model_validator

# Conversation flow:
chat_workflow/atomic_workflow.py:174  # AtomicWorkflow
chat_workflow/atomic_workflow.py:204  # process_turn() - main logic

# LLM integration:
chat_workflow/llm_interaction.py:44  # get_client() - provider setup
```

## Configuration

> Configuration is managed through config.json and environment variables. See the [User Guide](docs/user-guide.md) for detailed setup instructions.
```bash
# 1. Copy config template
# Edit config.json to set provider/model

# 2. Edit config.json to set provider/model
# 3. Set API key environment variable matching provider
export OPENROUTER_API_KEY=your-key-here  # if provider is "openrouter"
```

## Common Commands
```bash
# Run all unit tests (mocks only)
make test

# Check prompt changes work
python -c "from chat_workflow.atomic_workflow import AtomicWorkflow; from workflows.evaluation_criteria.models import EvaluationCriteria; from chat_workflow import AgentResponse; o=AtomicWorkflow(system_prompt='test', response_model=AgentResponse[EvaluationCriteria], max_turns=5, on_continue=lambda a: None, on_success=lambda a: None, on_failure=lambda a: Exception('x')); print(o.messages[0]['content'][:200])"

# Check configuration
python -c "from chat_workflow.config import config; print(config.provider, config.model)"
```

## When You're Stuck
1. **Business logic issue?** → Check `workflows/evaluation_criteria/models.py` validation
2. **Conversation flow problem?** → Check `chat_workflow/atomic_workflow.py` prompts & logic
3. **LLM integration failing?** → Check `chat_workflow/llm_interaction.py`
4. **Test failing?** → Check if it's a real API test needing keys