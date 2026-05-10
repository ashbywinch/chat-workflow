# Chat Workflow - Architecture Overview

This document is for developers contributing to the chat-workflow library. It describes the project's architecture, key files, and critical patterns to answer the question: "How is this library structured and how do the pieces fit together?"

### **Core Files (Read These First)**
1. `workflows/evaluation_criteria/models.py` - Data models & business rule validation
2. `chat_workflow/conversation_runtime.py` - Conversation orchestration & decorators  
3. `chat_workflow/llm_interaction.py` - LLM provider abstraction
4. `chat_workflow/exceptions.py` - Custom exception hierarchy

### **Configuration Files**
- `config.json` - LLM provider and model settings (REQUIRED)
- Environment variables - API keys (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
- `config.py` - Loads and manages configuration from `config.json`

## Critical Patterns

### 1. Prompt Design Philosophy
Prompts should focus on behavioral guidance (role, conversation strategy, interaction style) rather than data format. Instructor handles schema formatting automatically.

### 2. Validation Layers
```
User Input → LLM Response → Instructor → Pydantic Validation
```

### 3. Failure Modes
- **LLM-initiated**: Returns `action: "failure"` for unconstructive users
- **System-initiated**: `TurnLimitExceededError` when `max_turns` exceeded
- **Validation failure**: Custom exceptions for business rule violations

For a quick reference of common tasks and where to look, see the [Contributor Guide](docs/contributor-guide.md#common-tasks--where-to-look).
