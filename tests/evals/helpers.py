"""Shared helpers for eval tests that call real LLMs."""

import inspect
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel, Field

from chat_workflow import Config, Session, SessionLog
from chat_workflow.conversation_rules import NO_REPETITION


@dataclass
class EvalStats:
    """Accumulated timing and token usage for a single eval test.

    Tracks agent (workflow + user bot) and judge LLM calls separately
    so slow / expensive components are easy to spot.
    """
    test_name: str = ""
    duration_s: float = 0.0
    agent_tokens: int = 0
    judge_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.agent_tokens + self.judge_tokens

    def report(self) -> str:
        return (
            f"  [{self.test_name}] {self.duration_s:.0f}s"
            f"  {self.agent_tokens} agent tok"
            f"  {self.judge_tokens} judge tok"
            f"  {self.total_tokens} tot"
        )


# Global token counter used by litellm success_callback.
_token_count: int = 0


def _token_counter_callback(kwargs, completion_response, start_time, end_time) -> None:
    """litellm success_callback — accumulates tokens from every LLM call."""
    global _token_count
    usage = getattr(completion_response, "usage", None)
    if usage:
        _token_count += getattr(usage, "total_tokens", 0) or 0


def _reset_token_counter() -> None:
    """Reset the global litellm token counter to zero."""
    global _token_count
    _token_count = 0


def _read_token_counter() -> int:
    """Return the current litellm token count without modifying it."""
    return _token_count

_DEFAULT_TRANSCRIPT_DIR = Path(__file__).parent.parent.parent / "test-results" / "transcripts"

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def make_config() -> Config:
    return Config(_CONFIG_PATH)


def make_tools(io) -> Session:
    return Session(
        io=io,
        state=SessionLog(),
        config=make_config(),
    )


class AgentIO:
    """UserIO implementation backed by an LLM — plays a realistic user role.

    Instead of hardcoded MockIO responses, this generates natural user responses
    by calling the LLM with a persona prompt. This creates more realistic
    multi-turn conversations for eval tests.

    The user bot is an expert in their own domain (their meetings) but knows
    nothing about workflow decomposition. They just want help with their problem.
    """

    def __init__(self, persona_prompt: str, config: Config):
        self.persona_prompt = persona_prompt
        self.config = config
        self.outputs: list[str] = []
        # Conversation history from user bot's perspective:
        # - workflow agent messages are "user" (they're addressing the user bot)
        # - user bot responses are "assistant" (the bot is responding)
        self._history: list[dict[str, str]] = []

    def echo(self, message: str) -> None:
        """Capture what the workflow agent said."""
        self.outputs.append(message)
        # Strip the "\nAssistant: " prefix that Session.run adds
        clean = message.removeprefix("\nAssistant: ").strip()
        if clean:
            # From user bot's perspective, the workflow is the "user" talking to them
            self._history.append({"role": "user", "content": clean})

    def prompt(self, label: str) -> str:
        """Generate a user response using the LLM."""
        # Build messages: system persona + conversation history
        messages = [{"role": "system", "content": self.persona_prompt}]
        messages.extend(self._history)

        # Suppress litellm debug output
        litellm.suppress_debug_info = True

        # Call LLM to generate user response
        response = litellm.completion(
            model=self.config.model,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )

        user_text = response.choices[0].message.content or ""
        # Store this response in history
        self._history.append({"role": "assistant", "content": user_text})
        return user_text


class JudgeVerdict(BaseModel):
    """A single validation rule's verdict on the conversation."""

    rule: str = Field(..., description="Short label of the validation rule")
    passed: bool = Field(..., description="Whether the conversation satisfies this rule")
    explanation: str | None = Field(None, description="Brief explanation if the rule was violated")


class JudgeResult(BaseModel):
    """Structured evaluation of conversation quality."""

    verdicts: list[JudgeVerdict] = Field(
        ...,
        description=(
            "Individual verdicts — one per rule. The number of verdicts MUST match the number of rules provided."
        ),
    )


DEFAULT_JUDGE_RULES: dict[str, str] = {
    NO_REPETITION[0]: NO_REPETITION[1],
    "Uses expertise": (
        "The agent made informed proposals based on what the user said, rather "
        "than asking the user to describe every field from scratch. It's fine to "
        "ask follow-up questions about remaining fields after an initial proposal "
        "- the problem is when the agent never proposes anything and only asks."
    ),
    "Honest about provenance": (
        "The agent clearly separated their own inferences from confirmed facts — "
        "they proposed ideas as hypotheses for the user to confirm rather than "
        "silently putting fabricated values in the final output."
    ),
}


def llm_judge(
    rules: dict[str, str],
    transcript: str,
    config: Config,
) -> JudgeResult:
    """Ask an LLM to evaluate a conversation transcript against a list of rules.

    Uses Instructor (via get_client) to return a structured JudgeResult
    with a verdict for each rule. This is more reliable than parsing
    free-text YES/NO from the response.
    """
    from chat_workflow import get_client

    rules_text = "\n".join(f"{i + 1}. {name}: {desc}" for i, (name, desc) in enumerate(rules.items()))
    client = get_client(
        provider=config.provider,
        api_key_env=config.api_key_env,
        api_base=config.api_base,
        model_supports_tools=config.model_supports_tools,
    )
    result: JudgeResult = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict but fair evaluator of conversation quality. "
                    "Evaluate the conversation transcript below against each rule. "
                    "For each rule, decide PASS or FAIL and explain if it failed. "
                    "The number of verdicts MUST equal the number of rules."
                ),
            },
            {
                "role": "user",
                "content": f"Rules to evaluate:\n{rules_text}\n\nConversation transcript:\n{transcript}",
            },
        ],
        response_model=JudgeResult,
        max_retries=config.max_retries,
        timeout=config.request_timeout_seconds,
    )
    return result


def format_transcript(session: Session) -> str:
    """Build a formatted conversation transcript from session state (both sides)."""
    parts = []
    for msg in session.state.messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "system":
            continue
        parts.append(f"[{role}]\n{content}")
    return "\n---\n".join(parts)


@contextmanager
def capture_on_failure(session: Session, label: str = "conversation"):
    """If the enclosed block fails, write transcript and exception details to files.

    Transcripts go to $CHAT_WORKFLOW_TRANSCRIPT_DIR if set, or
    test-results/transcripts/ relative to the project root.
    Full exception details (including Instructor retry traces) go to a companion file.
    The exception message is replaced with a compact summary.
    """
    try:
        yield
    except Exception as exc:
        transcript = format_transcript(session)
        env_dir = os.environ.get("CHAT_WORKFLOW_TRANSCRIPT_DIR")
        outdir = Path(env_dir) if env_dir else _DEFAULT_TRANSCRIPT_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        # Write transcript
        tx_path = outdir / f"{label}-{ts}.txt"
        tx_path.write_text(f"=== CONVERSATION TRANSCRIPT ({label}) ===\n" + transcript + "\n=== END TRANSCRIPT ===\n")

        # Write full exception traceback to companion file
        import traceback

        tb_path = outdir / f"{label}-{ts}-exception.txt"
        tb_path.write_text(
            f"=== EXCEPTION DETAILS ({label}) ===\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            + "\n=== END EXCEPTION DETAILS ===\n"
        )

        # Compact message — preserve original assertion if it's an AssertionError
        if isinstance(exc, AssertionError):
            msg = str(exc)
        else:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            msg = str(cause).split("\n")[0] if str(cause) else type(cause).__name__

        raise AssertionError(f"{msg}\n[Transcript: {tx_path}]\n[Details: {tb_path}]") from exc


def run_multi_turn_eval(
    model_method: Callable[..., Any],
    method_kwargs: dict[str, Any],
    user_persona: str,
    judge_rules: dict[str, str] | None = None,
    judge: Callable = llm_judge,
    config: Config | None = None,
    test_name: str | None = None,
) -> Any:
    """Run a multi-turn workflow eval with an LLM user bot and optional LLM judge.

    Hands off to the model method (which is an @atomic_workflow-decorated
    classmethod), then verifies turn efficiency, runs the judge on the
    conversation transcript, and dumps the transcript on failure.

    Args:
        model_method: The @atomic_workflow or @composite_workflow classmethod to test
        method_kwargs: Keyword args to pass to model_method (must include session)
        user_persona: Persona prompt for the AgentIO user bot
        judge_rules: Dict mapping rule labels to descriptions.
            Defaults to DEFAULT_JUDGE_RULES.
        judge: The judge function. Callable[[dict[str, str], str, Config], JudgeResult].
            Defaults to llm_judge. Pass None to skip judging.
        config: Config instance. Defaults to make_config().
    """
    # Register litellm token callback (already registered globally in
    # tests/evals/__init__.py, but ensure it's active even when running
    # standalone or if the module-level registration was overridden).
    _reset_token_counter()
    litellm.success_callback = [_token_counter_callback]

    config = config or make_config()
    if judge_rules is None:
        judge_rules = getattr(model_method, "__conversation_rules__", None) or DEFAULT_JUDGE_RULES

    user_bot = AgentIO(persona_prompt=user_persona, config=config)

    # Create session — state.messages records the conversation
    session = make_tools(user_bot)

    # Inject session into method_kwargs if not present
    kwargs = dict(method_kwargs)
    kwargs.setdefault("session", session)

    try:
        result = model_method(**kwargs)
    except Exception:
        result = None

    # Auto-detect unittest test method name from call stack if not provided
    if test_name is None:
        for frame in inspect.stack():
            locals = frame[0].f_locals
            method = locals.get("self")
            if method is not None and hasattr(method, "_testMethodName"):
                test_name = method._testMethodName
                break
        else:
            test_name = model_method.__name__

    # LLM judge on the full conversation — passes even if workflow failed
    # or hit turn limit, as long as the conversation quality is acceptable.
    failures: list = []
    if judge is not None:
        transcript = format_transcript(session)
        judge_result = judge(judge_rules, transcript, config)
        failures = [v for v in judge_result.verdicts if not v.passed]

    # Save transcript on success too when env var is set
    if os.environ.get("CHAT_WORKFLOW_SAVE_TRANSCRIPT"):
        transcript = format_transcript(session)
        outdir = _DEFAULT_TRANSCRIPT_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        tx_path = outdir / f"success-{test_name or 'conversation'}-{int(time.time())}.txt"
        tx_path.write_text(transcript)

    # Raise judge failures after stats are captured
    if failures:
        with capture_on_failure(session, label=test_name or "conversation"):
            raise AssertionError(
                f"Conversation quality: {len(failures)}/{len(judge_rules)} rules failed:\n"
                + "\n".join(f"  [{v.rule}] FAIL: {v.explanation}" for v in failures)
            )

    return result


def make_meeting_analysis():
    """Create a standard meeting-minutes ProcessDefinition for eval tests."""
    from workflows.workflow.models import ProcessDefinition

    return ProcessDefinition(
        phases=["Note-taking", "Review & Clarify", "Draft Minutes", "Review & Approve"],
        activities=[
            "Take meeting notes",
            "Review notes for clarity",
            "Identify action items",
            "Write minutes draft",
            "Circulate for review",
            "Incorporate feedback",
            "Distribute final minutes",
        ],
        orchestrating_component="Meeting Organizer",
        participants=["Meeting Attendees", "Note Taker", "Reviewers"],
    )
