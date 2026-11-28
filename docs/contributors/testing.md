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
- **QA scenarios catch integration issues** — unit tests and evals test in isolation; QA scenarios verify the system works end-to-end (e.g., simulate a model retry → verify the companion file is created and no traceback reaches the user).

> One rule of thumb: if you changed only prompt text, write an eval. If you changed only code, write a unit test. If you changed both, write both.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                      # Shared fixtures: timeout, FakeConfig, FunctionModel
├── unit/
│   ├── test_models.py              # Pydantic model validation (no mock)
│   ├── test_orchestrator_logic.py  # Orchestrator logic (mock _call_llm)
│   └── test_conversation.py        # the model call (FunctionModel)
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

### 2. The model call (`FunctionModel`)
``Conversation`` takes a ``model_client`` seam. A PydanticAI ``FunctionModel``
scripts a whole turn without a network: it receives the message list and the
agent info and returns a ``ModelResponse``. The answer itself is an ``answer``
tool call, so a scripted response includes that call. ``tests/conftest.py``
provides ``make_conversation`` for the common case.

### 3. Real API Evals (no mocking)
Evals call a real LLM to verify prompt quality. They require `config.json` with a valid provider
and the gateway token (`OPENAI_API_KEY`) plus `OPENAI_BASE_URL`. One-shot
evals also work against a direct provider by switching the active preset and
setting that provider's key. They **fail**, not skip, when credentials are
missing — that's intentional.

> ⚠️ **Security**: See the [agent rules](../../AGENTS.md#rules) for how to safely check
> that environment variables are set without exposing their values.

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
To test an actual `@atomic_workflow_step` decorated method with its real docstring, use `AgentIO` — an
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
make evals              # Full suite (real API, ~2h, 4h wrapper in CI)
make evals-test TEST=...  # One targeted eval (module/class/test path)
make evals-smoke        # Quick framework-level check (test_real_api + debug_streaming, ~5 min)
make evals-incremental  # Change-aware subset (auto-detects affected evals via code-review-graph, 1800s timeout)
make evals-verbose      # Same with verbose output
CHAT_WORKFLOW_DEBUG=1 make evals   # Evals with LLM request/response tracing
make test-all           # Unit tests + evals
make lint               # ruff check
```

### Cost & time tracking

Set `CHAT_WORKFLOW_EVAL_REPORT=1` to capture per-test timing and token counts to
`test-results/eval-report.txt`:

```bash
CHAT_WORKFLOW_EVAL_REPORT=1 make evals 2>&1 | tee .sisyphus/evidence/run-$(date +%s).txt
```

This is useful for identifying slow or expensive tests, and for comparing
the cost impact of prompt changes.

## Eval Design Principles

### Session turn budget

An eval's `max_turns` is the whole session's runaway guard, passed by
`run_multi_turn_eval` to `Session.budget`. It is not a per-step allowance and
no step can shorten it, so it must fit the whole conversation the eval drives —
including every child step in the session tree.

Size the budget from the conversation, not from a table: start from the
framework default and lower it only when the transcript shows the eval does
not need the room. A budget that cuts a healthy conversation short turns a
prompt defect into a turn-limit failure, and a budget that is too generous only
costs time when the workflow is already looping. Do not use the budget to force
a step to finish; the ending is the step's success or failure.

### Front-loaded user personas

To reduce turn counts in end-to-end evals without degrading quality, design
user personas that volunteer comprehensive information. The key pattern:

```
"When the assistant asks about one aspect, volunteer ALL related details
immediately — don't wait for separate follow-ups about each field."
```

This prevents the agent from needing N separate turns to gather N fields.
Instead, one turn about "what happened in the meeting" triggers a response
that covers attendees, decisions, action items, dates, and budget.

### Safety: preventing infinite token burning

Eval suites call real LLM APIs and can burn tokens indefinitely if
something goes wrong. **Never rely on prompts or correct code to prevent
infinite loops** — the whole point of CI is to catch when code is wrong.

Use multiple layers of protection (belt and braces):

1. **Per-test timeout** — `@timeout(N)` decorator (SIGALRM) on every eval test.
   Each test has its own generous timeout (120s for component evals, 300s for
   end-to-end pipeline evals).

2. **Session turn budget** — `run_multi_turn_eval`'s `max_turns` constructs a
   `TurnBudget` for the whole session; the framework default is the guard for
   a run that does not set one. Every step in the session tree takes from that
   one counter, so a runaway conversation still stops.

3. **Overall suite timeout** — `scripts/run_with_timeout.py` wraps the entire
   eval suite with a hard kill (14400s by default for `make evals` and
   `make evals-incremental`, the same guard CI uses; the full suite takes
   ~2h). If the suite exceeds this, the process is killed and the remaining
   tests are reported as incomplete.

These three layers ensure that a buggy workflow or runaway LLM can never
burn tokens indefinitely, even if all three fail partially.

### Cost optimization

- **Use `make evals-test TEST=...` for one targeted test** while iterating.
  Never run the full suite after every change.
- **Use `make evals-incremental` for a change-aware subset** — it detects
  which tests are affected by your changes via the code-review-graph
  dependency analysis. This saves time and tokens on iterative work.
- **CI always runs `make evals` (full suite)** — see ``.github/workflows/tests.yml``.
  Do not rely on the incremental runner for CI; it's a local optimization.
  The full suite is a required PR gate, not a weekly/manual run.
- **Reuse existing output** — if you haven't changed code, don't re-run
  evals. Check `make evals-incremental` output or the CI log from the PR.
- **Capture output with tee** — always pipe eval runs through `tee` to a
  file so you don't need to re-run just to read results.
- **Fix the test, not the model** — eval failures nearly always point to
  prompt improvements, not model deficiencies. See "Prompt Improvement
  Mindset" below.

### Eval classification: single-workflow vs composite

Evals live in two subdirectories under ``tests/evals/``:

- **``single/``** — one ``@atomic_workflow_step`` method, one user persona, one
  LLM judge. These are the cheapest to run and best for rapid iteration
  on prompt changes (like unit tests, but for prompts).
- **``composite/``** — run full pipelines combining multiple workflow
  phases (``@composite_workflow_step``, end-to-end generation, debug
  streaming). These are integration tests.

**Guideline**: When iterating on a workflow prompt, run the relevant
``single/`` eval first (cheap, fast). Run ``composite/`` evals only
once the single-workflow tests pass, before pushing.

The cost report (``scripts/eval_report.py``) automatically groups tests
by their file location, making it easy to spot the most expensive evals.

### Provider selection and the model path

There is one model path: `Conversation.send()` in
`chat_workflow/conversation.py` calls PydanticAI, which calls an
OpenAI-compatible endpoint. The repo's `config.json` names the model and the
key environment variable; `OPENAI_BASE_URL` points at the Cloudflare AI
Gateway compat endpoint. The eval helper fails loudly when the base URL is
missing rather than silently calling api.openai.com.

A step's answer is a forced `answer` tool call
(`ToolOutput(..., strict=True)`). `response_format` is not used: the endpoint
accepts it and ignores it, while a tool's schema is enforced on the same
route. The tool's schema is built from the step's result type, so a
discriminated `AgentResponse` is the protocol.

`temperature`, `max_retries` and `request_timeout_seconds` live under `llm` in
`config.json`. The eval helper overrides the gateway cache header and the
timeout for a cold reasoning route; production keeps the configured values.

#### What model to use

OpenCode Go (``$10``/month subscription):

| Model | Input/1M | Output/1M | Speed (p50) | Notes |
|---|---|---|---|---|
| DeepSeek V4 Flash | \$0.14 | \$0.28 | 3.72s / 42 tok/s | Best value, good for coding |
| MiMo-V2.5 | \$0.14 | \$0.28 | 3.24s / 35 tok/s | Same price, similar speed |
| MiMo-V2.5-Pro | \$1.74 | \$3.48 | — | 12× more expensive |
| Kimi K2.5 | \$0.60 | \$3.00 | — | Good quality, higher cost |
| GLM-5 | \$1.00 | \$3.20 | — | High quality, expensive |

The cheapest models are DeepSeek V4 Flash and MiMo-V2.5 at the same price.
Both go through the OpenCode Go proxy which adds latency — expect per-test
eval times **5-10× slower** than the same model on a direct provider like
OpenRouter.

For faster eval runs with comparable pricing, use the ``openrouter`` preset
which routes through OpenRouter's lower-latency endpoints.

## Test Coverage Goals

- **Models**: 100% (validation, business rules)
- **Orchestrator logic**: 100% (turn management, action handling)
- **LLM interaction**: 100% (retry logic, error handling)
- **Integration / Evals**: Critical paths only (real API interaction)
  - Every workflow's `generate_from_chat` prompt should have at least one multi-turn eval
  - Write evals that reproduce known failure modes and verify they're fixed

## Prompt Improvement Mindset

An eval failure is diagnosed by the rules in
[Testing Standards](../testing-standards.md#an-eval-failure-is-a-defect), not
by re-running until green. The patterns below are how to improve the prompt
once the failure has been diagnosed.

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

Read the failed turn's reasoning in the `.json` before changing anything:
name the instruction the model followed and the instruction it ignored.

1. Read the judge's verdict — it tells you *which* rule was violated and *why*.
2. If "No repetition" failed: did the agent ignore or re-ask for information the user already gave? If yes, add guidance to listen and build on answers. If the follow-up was genuinely needed (e.g., user gave a vague answer), the judge prompt may need refining instead.
3. If "Uses expertise" failed: the agent is asking the user to fill out a form. Add language that tells it to propose, infer, and suggest — "propose the details yourself" — and give a concrete example.
4. If "Honest about provenance" failed: the agent is hallucinating. Add language that clearly separates proposals from confirmed facts.
5. Check the final-turn contract: a success with no result, or a confirmation
   question, means the error the model received did not say what to do at
   that point. Improve the `FinalAnswer` message so it says what to return.
6. Check the turn budget: the designed flow (gather → draft → confirm →
   result, plus a late correction) must fit. A final-turn failure often means
   the ceiling is too tight, not that the model is incapable.
7. Fix the contract, prompt/context or budget. Follow the
   [prompt and framework-change rules](../testing-standards.md): a prompt
   changes as a whole, and a framework change is a last resort that must be
   agreed with the user. A re-run confirms a change or tests a named
   nondeterminism hypothesis.
8. Transcripts are saved to `test-results/transcripts/`: the `.json` file is
   the canonical record (typed events, result, judge verdicts with evidence and
   fix, metadata); the `.txt` file is the rendered human view; the
   `-exception.txt` file has the full traceback.
### Resuming a failed eval

The workflow pipeline writes its component specs to
`.sisyphus/checkpoints/workflow_pipeline-latest.json` before it asserts, so a
component failure can be re-run without repeating the pipeline:

```bash
RESUME_SPECS=.sisyphus/checkpoints/workflow_pipeline-latest.json \
RESUME_NAMES="Approval Record" \
make evals-test TEST=tests.evals.composite.test_resume_component_generation
```

The resume test skips unless `RESUME_SPECS` is set, so the normal suite is
unaffected. Resume one component at a time; the test's budget covers one. This
is for iteration — the full pipeline remains the gate. The general pattern is
in `skill://eval-resume`.
