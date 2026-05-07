# Chat Workflow - Architecture Overview

## Quick Navigation Guide

This project uses uv to manage dependencies
There is a Makefile. All git workflows go through the Makefile.

### **Core Files (Read These First)**
1. `workflows/evaluation_criteria/models.py` - Data models & business rule validation
2. `chat_workflow/conversation_runtime.py` - Conversation orchestration & decorators  
3. `chat_workflow/llm_interaction.py` - LLM provider abstraction
4. `chat_workflow/exceptions.py` - Custom exception hierarchy

### **Configuration Files**
- `config.json` - LLM provider and model settings (REQUIRED)
- Environment variables - API keys (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
- `config.py` - Loads and manages configuration from `config.json`

### **Key Concepts**
- **High usability for workflow authors**: Authors of workflows (such as EvaluationCriteria) must be able to easily figure out the library and create extremely readable/understandable workflows with an absolute minimum of boilerplate
- **Structured outputs via Instructor**: LLM returns Pydantic objects
- **Validation-first**: Business rules in Pydantic models, never in prompts
- **Multi-turn conversation**: Stateful orchestrator manages dialogue flow
- **Configurable failure modes**: LLM can fail OR system can hit turn limits
- **Dual configuration**: Provider/model in `config.json`, API keys in environment
- **Configuration at the edge**: Configuration (file paths, environment variables, etc) must only ever be set/read at the perimeter, i.e. in a CLI app, a test setup, etc.
- **Fail fast**: We are never backwards compatible. If something is configured or set up incorrectly we fail fast instead of using defaults.
- All error messages shown to the user or the workflow author should be user-friendly

## File Responsibilities

### `chat_workflow/conversation_runtime.py` - Conversation Logic
```python
# Core class: StructuredConversationOrchestrator
# - Manages turn state (max_turns configurable)
# - Receives system prompt from @chat decorator
# - Three outcomes: continue/success/failure
```

### `llm_interaction.py` - LLM Abstraction
```python
# Unified client for multiple providers
get_client()  # Returns instructor-patched client
# Supports: OpenAI, Google, OpenRouter, etc.
```

### `config.py` - Configuration Management
```python
# Singleton configuration manager
config = Config()  # Reads ONLY from config.json
# Provides: provider, model, temperature, max_retries
# Note: API keys come from environment variables, not config.json
```

## Critical Patterns

### 1. Prompt Design Philosophy
See [AGENTS.md](AGENTS.md#prompt-design-rules) for the full prompt design rules.

### 2. Validation Layers
```
User Input → LLM Response → Instructor → Pydantic Validation
```

### 3. Failure Modes
- **LLM-initiated**: Returns `action: "failure"` for unconstructive users
- **System-initiated**: `TurnLimitExceededError` when `max_turns` exceeded
- **Validation failure**: Custom exceptions for business rule violations

## Common Tasks & Where to Look

| Task | Primary File | Key Function/Method |
|------|--------------|---------------------|
| Add business rule | `workflows/evaluation_criteria/models.py` | `EvaluationCriteria.validate_business_rules()` |
| Modify generic prompt | `chat_workflow/conversation_runtime.py` | `@chat` decorator system prompt |
| Add test for new feature | `tests/unit/` | Follow existing test patterns |
| Add eval for new feature | `tests/evals/` | Follow existing eval patterns |

## Testing Strategy
- **Unit tests**: `tests/unit/` - Mock LLM, test logic
- **Eval tests**: `tests/evals/` - Real API calls (require keys)
- **Key principle**: Evals fail without API keys (exposes setup issues)

## Quick Start for Common Changes

### Modify Conversation Flow  
1. Edit `@chat`-decorated function docstrings in `workflows/evaluation_criteria/evaluation_criteria.py`
2. Check `StructuredConversationOrchestrator.process_turn()` logic in `conversation_runtime.py`
3. Update `tests/unit/test_orchestrator_logic.py`

### Add LLM Provider
1. Update `chat_workflow/llm_interaction.py` `get_client()`
2. Add provider configuration handling

