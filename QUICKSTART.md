# Quick Start for New Contributors

## 30-Second Overview
- **Goal**: Generate evaluation criteria via LLM conversation
- **Core**: `ConversationOrchestrator` manages multi-turn dialogue  
- **Output**: `EvaluationCriteria` Pydantic model (validated)
- **Key rule**: Must include "budget" criterion, ≥2 total criteria

## First 5 Files to Read
1. `evaluation_criteria/models.py` - Data structures & validation rules
2. `prompt_core/conversation_runtime.py` - Conversation orchestration & decorators
3. `tests/unit/test_models.py` - See business rule tests
4. `tests/unit/test_orchestrator_logic.py` - See conversation flow tests
5. `spec.md` - Product requirements

## Critical Code Locations
```python
# Business rules (MUST maintain):
evaluation_criteria/models.py:44  # validate_business_rules() model_validator

# Conversation flow:
prompt_core/conversation_runtime.py:174  # StructuredConversationOrchestrator
prompt_core/conversation_runtime.py:204  # process_turn() - main logic

# LLM integration:
prompt_core/llm_interaction.py:44  # get_client() - provider setup
```

## Configuration

> Prompt design rules are documented in [AGENTS.md](AGENTS.md#prompt-design-rules).
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
python -c "from prompt_core.conversation_runtime import StructuredConversationOrchestrator; from evaluation_criteria.models import EvaluationCriteria; from prompt_core import ConversationAction; o=StructuredConversationOrchestrator(system_prompt='test', response_model=ConversationAction[EvaluationCriteria], max_turns=5, on_continue=lambda a: None, on_success=lambda a: None, on_failure=lambda a: Exception('x')); print(o.messages[0]['content'][:200])"

# Check configuration
python -c "from prompt_core.config import config; print(config.provider, config.model)"
```

## When You're Stuck
1. **Business logic issue?** → Check `evaluation_criteria/models.py` validation
2. **Conversation flow problem?** → Check `prompt_core/conversation_runtime.py` prompts & logic
3. **LLM integration failing?** → Check `prompt_core/llm_interaction.py`
4. **Test failing?** → Check if it's a real API test needing keys