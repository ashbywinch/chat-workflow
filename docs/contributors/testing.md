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

## Test Strategy Rules

These rules govern when and how tests must be written for any change:

| Type of Change | Required Test |
|---|---|
| Code changes only | Failing unit test first (TDD: red → green → refactor) |
| Prompt text changes only | Failing eval first (eval-first: write the eval, watch it fail, fix the prompt) |
| Both code and prompt changes | Both — failing unit test AND failing eval first |
| Any of the above | Agent-executed QA scenarios are mandatory for EVERY task, regardless |

### Why This Matters

- **Code without a unit test is untested logic** — the code path may never be exercised.
- **Prompt text without an eval is a guess** — you won't know if the change improves or degrades conversation quality until you run a real LLM against it.
- **QA scenarios catch integration issues** — unit tests and evals test in isolation; QA scenarios verify the system works end-to-end (e.g., simulate an InstructorRetryException → verify companion file created + no traceback visible).

> One rule of thumb: if you changed only prompt text, write an eval. If you changed only code, write a unit test. If you changed both, write both.

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
class TestProcessDefinitionEval(unittest.TestCase):
    @timeout(30)
    def test_llm_produces_process_definition(self):
        from workflows.workflow.models import ProcessDefinition

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt="You are a Business Process Analyst...",
                response_model=AgentResponse[ProcessDefinition],
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
            self.assertIsInstance(result.result, ProcessDefinition)
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

analysis = generate_from_chat(
    session=session,
)

# Structural assertions (Pydantic validates the output)
self.assertIsInstance(analysis, ProcessDefinition)
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
make evals              # Full suite (real API, ~90-300s)
make evals-smoke        # Quick framework-level check (test_real_api + debug_streaming, ~80s)
make evals-incremental  # Change-aware subset (auto-detects affected evals via code-review-graph)
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

## Prompt Improvement Mindset

When an eval fails, it's tempting to blame the cheap model. **Resist that instinct.** A core purpose of this library is to build agents that work well even with small, fast, cheap models. An eval failure is almost always a signal that the prompt can be improved.

### What good prompts look like

The best prompts for multi-turn conversation workflows share a few patterns:

1. **They tell the agent *how* to think, not just *what* to produce.**
   A prompt that lists fields like "produce: consumer, format, success_criteria..." trains the agent to follow a checklist. A prompt that says "you're an expert — use what the user tells you to fill in the details yourself" trains the agent to listen and propose.

2. **They describe the conversational rhythm with concrete examples.**
   "Synthesize what they've said and share your understanding" is abstract. "For example: 'From what you've described, I'm seeing three phases... Does that match your understanding?'" gives the model a concrete pattern to follow.

3. **They tell the agent what to do when the user gives a detailed answer.**
   Without this guidance, agents tend to ignore verbose user responses and continue their pre-planned script. Telling them to "acknowledge and build on what the user just said, rather than asking for it again" prevents repetition loops.

4. **They frame the task around the user's domain, not the data model.**
   "Identify consumer, format, and success criteria" makes the agent think about fields. "Help the user understand who uses each output and what makes it good" makes the agent think about the user's problem. The data model is the same; the conversation quality is very different.

### When debugging a failing eval

1. Read the judge's verdict — it tells you *which* rule was violated and *why*.
2. If "No repetition" failed: did the agent ignore or re-ask for information the user already gave? If yes, add guidance to listen and build on answers. If the follow-up was genuinely needed (e.g., user gave a vague answer), the judge prompt may need refining instead.
3. If "Uses expertise" failed: the agent is asking the user to fill out a form. Add language that tells it to propose, infer, and suggest — "propose the details yourself" — and give a concrete example.
4. If "Honest about provenance" failed: the agent is hallucinating. Add language that clearly separates proposals from confirmed facts.
5. Rerun the failing test only — never the full suite until you're ready to verify.
6. Transcripts are saved to `test-results/transcripts/` for debugging.