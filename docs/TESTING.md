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
│   ├── test_conversation.py        # the model path (scripted FunctionModel)
│   └── test_transcript.py          # the transcript projection
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
        Continue(intent=AgentIntent.CONTINUE, message="Question 1"),
        Continue(intent=AgentIntent.CONTINUE, message="Question 2"),
        Success[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=valid_criteria),
    ]
    mock_call_llm.side_effect = responses
    result1 = orchestrator.process_turn("Hello")
    assert not result1.is_complete
```

### 2. The model path (script a `FunctionModel`)
```python
# tests/conftest.py builds a Conversation on a scripted model.
conversation = make_conversation([{"intent": "continue", "message": "Test"}])
orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(conversation=conversation))
action = orchestrator.process_turn("Hello")
```
The script is the model: no network, no provider SDK, and the exact responses a
gateway has been seen to produce (blank, token-limit, wrong envelope). For the
step's turn rules alone, patch `_call_llm`.

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