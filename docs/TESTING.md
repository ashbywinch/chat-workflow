# Testing Strategy

This guide is for developers contributing to the chat-workflow library. It describes the testing strategy and patterns to answer the question: "How should I test changes to this library?"

## Philosophy

1. **Fail fast with exceptions** — code raises, doesn't return failure objects
2. **Clear separation** — unit tests (mocked) vs evals (real API)
3. **Infrastructure exposure** — evals **fail (not skip)** when API keys missing
4. **Proper mocking** — test logic with mocks, test prompts with real API

## Test Structure

```
tests/
├── unit/
│   ├── test_models.py              # Pydantic model validation (no mock)
│   ├── test_orchestrator_logic.py  # Orchestrator logic (mock _call_llm)
│   └── test_llm_interaction.py     # LLM interaction (mock get_client)
└── evals/
    └── test_real_api.py            # Real API tests (require config.json + API key)
```

## Key Patterns

### 1. Orchestrator Logic (mock `_call_llm`)
```python
@patch.object(AtomicWorkflow, '_call_llm')
def test_multi_turn_conversation(self, mock_call_llm):
    orchestrator = AtomicWorkflow(
        system_prompt="Test", response_model=AgentResponse[EvaluationCriteria],
        max_turns=5, initial_messages=None,
        on_continue=..., on_success=..., on_failure=...,
    )
    responses = [
        AgentResponse(intent=AgentIntent.CONTINUE, message="Question 1"),
        AgentResponse(intent=AgentIntent.CONTINUE, message="Question 2"),
        AgentResponse(intent=AgentIntent.SUCCESS, result=valid_criteria),
    ]
    mock_call_llm.side_effect = responses
    result1 = orchestrator.process_turn("Hello")
    assert not result1.is_complete
```

### 2. LLM Interaction (mock `get_client`)
```python
@patch('chat_workflow.llm_interaction.get_client')
def test_call_llm_success(self, mock_get_client):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = AgentResponse[...](
        intent=AgentIntent.CONTINUE, message="Test"
    )
    mock_get_client.return_value = mock_client
    action = self.orchestrator._call_llm()
```

### 3. Real API Evals (no mocking)
- Verify prompts generate valid structured output from real LLMs
- **Will fail** without valid API keys — intentional

## Commands

```bash
make test              # Unit tests (no API key, ~0.01s)
make test-verbose      # Same with verbose output
make evals              # Real-API evals (~90s, requires API key)
make evals-verbose      # Same with verbose output
make lint               # black --check + ruff check
```

## Test Coverage Goals

- **Models**: 100% (validation, business rules)
- **Orchestrator logic**: 100% (turn management, action handling)
- **LLM interaction**: 100% (retry logic, error handling)
- **Integration**: Critical paths only (real API interaction)