# Testing Strategy

This guide is for developers contributing to the chat-workflow library. It describes the testing
strategy and patterns to answer the question: "How should I test changes to this library?"

## Philosophy

1. **Fail fast with exceptions** — code raises, doesn't return failure objects
2. **Clear separation** — unit tests (mocked) vs evals (real API). If your test doesn't call a real
   LLM API, it is a test, not an eval. Don't call it an eval.
3. **Infrastructure exposure** — evals **fail (not skip)** when API keys missing
4. **Proper mocking** — test logic with mocks, test prompts with real API
5. **Test the output with Pydantic, test the conversation with LLM judges** — structural validity
   belongs in assertions, but conversation quality (did the agent loop? repeat itself? use jargon?)
   is best assessed by an LLM judge reading the transcript.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                      # Shared fixtures: timeout, FakeConfig, MockInstructorClient
├── unit/
│   ├── test_models.py              # Pydantic model validation (no mock)
│   ├── test_orchestrator_logic.py  # Orchestrator logic (mock _call_llm)
│   └── test_llm_interaction.py     # LLM interaction (mock get_client)
└── evals/
    ├── __init__.py
    ├── helpers.py                  # AgentIO, llm_judge, make_config, make_tools
    ├── test_real_api.py            # Real API tests (require API key)
    └── test_workflow_evals.py      # Workflow prompt quality evals (require API key)
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
Evals call a real LLM to verify prompt quality. They require `config.json` with a valid provider
and the corresponding `OPENROUTER_API_KEY` (or similar) environment variable. They **fail**, not
skip, when keys are missing — that's intentional.

#### 3a. One-Shot Evals (hardcoded prompt)
Test that a given prompt + user input produces valid structured output in a single turn:

```python
class TestProcessAnalysisEval(unittest.TestCase):
    @timeout(30)
    def test_llm_produces_process_analysis(self):
        from workflows.workflow.models import ProcessAnalysis

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt="You are a Business Process Analyst...",
                response_model=AgentResponse[ProcessAnalysis],
                max_turns=3,
                ...
                initial_messages=[{"role": "user", "content": "Customer places an order..."}],
                on_continue=lambda action: TurnResult[...].continuing(action.message or ""),
                on_success=lambda action: TurnResult[...].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or ""),
            )
        )
        result = orchestrator.process_turn("Please analyze...")
        if result.result:
            self.assertIsInstance(result.result, ProcessAnalysis)
            self.assertTrue(len(result.result.phases) >= 1)
```

Use one-shot evals to verify models can be populated, but understand they don't test the actual
workflow's decorated docstring — they test a simplified version.

#### 3b. Multi-Turn Evals with AgentIO (LLM-powered user bot)
To test an actual `@atomic_workflow` decorated method with its real docstring, use `AgentIO` — an
LLM-powered `UserIO` implementation that plays a realistic user role:

```python
from tests.evals.helpers import AgentIO, make_tools

user_persona = (
    "You are a busy professional who needs help creating a process for "
    "turning meeting notes into minutes. You know your meetings inside out "
    "but know nothing about workflow jargon."
)

user_bot = AgentIO(persona_prompt=user_persona, config=_CONFIG)
session = make_tools(user_bot)

analysis = ProcessAnalysis.generate_from_chat(
    process_description="Writing up my sketchy meeting notes...",
    session=session,
)

# Structural assertions (Pydantic validates the output)
self.assertIsInstance(analysis, ProcessAnalysis)
self.assertGreaterEqual(len(analysis.phases), 1)

# Turn efficiency (key regression guard for Socratic loops)
self.assertLess(session.state.turn_count, 10)
```

**Why AgentIO instead of MockIO?** Hardcoded `MockIO` response lists break the moment the LLM asks
a slightly different question than expected. AgentIO adapts — the bot responds naturally to whatever
the workflow agent says, making the eval more realistic and less brittle.

**Persona design tips:**
- The user is an expert in **their domain** but knows nothing about workflows
- Personify the specific problem the workflow is designed to solve
- Include behavioral cues like "be patient but don't repeat yourself"
- Clarify the frame: "they're designing a process, not doing actual work right now"

#### 3c. LLM Judges for Non-Deterministic Criteria
Use `llm_judge()` to evaluate aspects that can't be captured by Pydantic validation or simple
assertions — especially conversation quality:

```python
from tests.evals.helpers import llm_judge

# Build the conversation transcript from AgentIO's history
transcript = "\n---\n".join(
    msg["content"]
    for msg in user_bot._history
    if msg["role"] == "user"  # workflow messages stored as "user" from bot's perspective
)

is_good, reason = llm_judge(
    "Evaluate this conversation transcript. Did the analyst:\n"
    "- Synthesize the user's description into a coherent proposal?\n"
    "- Avoid repeating the same question?\n"
    "- Use plain language the user can understand?\n\n"
    "Answer YES for good conversation quality, NO if the agent was stuck in a loop.",
    transcript,
    _CONFIG,
)
self.assertTrue(is_good, f"Conversation quality issue:\n{reason}")
```

**What to judge (and what not to):**
- **DO judge**: conversation transcript — did the agent synthesize, loop, repeat itself, use jargon?
- **DO NOT judge**: the structured output (Pydantic already validates fields)
- **DO NOT use keyword matching** for conversation quality — it's too brittle. Use an LLM judge.

**Tips for writing judge prompts:**
- Frame the question as concrete behavioral criteria ("did the agent do X?")
- Ask for YES/NO on its own line, then reasoning — the reasoning helps diagnose failures
- Use temperature=0.0 for consistent judgments
- Max tokens 200 is usually enough for verdict + brief reasoning

## Commands

```bash
make test              # Unit tests (no API key, ~0.01s)
make test-verbose      # Same with verbose output
make evals              # Real-API evals (~90s, requires API key)
make evals-verbose      # Same with verbose output
make evals-debug        # Evals with LLM request/response tracing
make test-all           # Unit tests + evals
make lint               # ruff check
```

## Test Coverage Goals

- **Models**: 100% (validation, business rules)
- **Orchestrator logic**: 100% (turn management, action handling)
- **LLM interaction**: 100% (retry logic, error handling)
- **Integration / Evals**: Critical paths only (real API interaction)
  - Every workflow's `generate_from_chat` prompt should have at least one multi-turn eval
  - Write evals that reproduce known failure modes and verify they're fixed