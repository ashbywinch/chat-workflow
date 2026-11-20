"""Shared helpers for eval tests that call real LLMs."""

import inspect
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelRequest

from chat_workflow import (
    Config,
    Conversation,
    Session,
    SessionComponent,
    SessionLog,
)
from chat_workflow.conversation_rules import NO_REPETITION
from chat_workflow.transcript import entries, render


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


_DEFAULT_TRANSCRIPT_DIR = Path(__file__).parent.parent.parent / "test-results" / "transcripts"

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"

# The gateway caches a completion by request body and replays it without
# calling the model. An eval that is served a stored answer measures the cache,
# not the prompt. A cold reasoning route also needs more than the production
# timeout, so the eval ceiling is explicit.
_EVAL_OVERRIDES = {
    "llm": {
        "extra_headers": {"cf-aig-skip-cache": "true"},
        "request_timeout_seconds": 300,
    }
}


def make_config() -> Config:
    """The house config, with the gateway cache bypassed and room for a cold call."""
    config = Config(_CONFIG_PATH, overrides=_EVAL_OVERRIDES)
    if config.provider == "openai" and not config.api_base:
        raise RuntimeError(
            "OPENAI_BASE_URL is not set. Evals must route through Cloudflare AI "
            "Gateway — run `make setup` to create .env from .env.example."
        )
    return config


def make_tools(io, config: Config | None = None) -> Session:
    return Session(
        io=io,
        state=SessionLog(),
        config=config or make_config(),
    )


def eval_timeout(config: Config) -> int:
    """A ceiling for one eval, derived from the budget the workflow may spend."""
    return int(config.request_timeout_seconds * (config.max_retries + 1)) + 15


_DEFAULT_TRANSCRIPT_DIR = Path(__file__).parent.parent.parent / "test-results" / "transcripts"

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def _looks_garbled(text: str) -> bool:
    """True for a completion that is not a sentence a person would write."""
    words = text.split()
    if not words:
        return True
    stars = text.count("*")
    if stars > 8:
        return True
    if text.count('"') % 2:
        return True
    if any(character in "（）［］｛｝" for character in text):
        return True
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        return True
    weird = sum(
        1
        for word in words
        if not any(character.isalnum() for character in word) or len(word) > 40
    )
    return len(words) >= 5 and weird / len(words) > 0.4


class AgentIO:
    """UserIO implementation backed by an LLM — plays a realistic user role.

    The user bot is a :class:`~chat_workflow.conversation.Conversation` with a
    persona. Workflow messages are fed to it as user turns; its replies are
    assistant turns. It is a model call like every other one, through the same
    gateway and the same transcript.
    """

    def __init__(self, persona_prompt: str, config: Config):
        self.persona_prompt = (
            "You are role-playing the user in a conversation. Stay in character "
            "as the person described below. You are the person asking for help, "
            "never the helper. Never offer to document or plan anything for the "
            "other person, never ask what they produce, and never explain what "
            "you are here for. Never say you are an AI, a model, or an assistant, "
            "and never break character. If a question seems addressed to the "
            "wrong person, answer it as yourself anyway. Reply in English. If "
            "the other person writes in another language, say 'Please continue "
            "in English' and answer in English.\n\n" + persona_prompt
        )
        self.config = config
        self.outputs: list[str] = []
        self._persona = persona_prompt
        self._conversation = Conversation(
            model=config.model,
            provider=config.provider,
            instructions=self.persona_prompt,
            output_type=str,
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            output_retries=config.max_retries,
            transport_retries=config.max_retries,
            timeout_seconds=config.request_timeout_seconds,
            extra_headers=config.extra_headers,
            temperature=0.7,
        )

    def echo(self, message: str) -> None:
        """Capture what the workflow agent said."""
        self.outputs.append(message)
        clean = message.removeprefix("\nAssistant: ").strip()
        if clean:
            self._conversation.messages.append(
                ModelRequest.user_text_prompt(
                    "Remember who you are. This is your role, not the "
                    f"helper's:\n{self._persona}\n\n"
                    "The helper said:\n"
                    f"{clean}\n\n"
                    "Write your next message as that person. Do not offer to "
                    "help, do not ask the helper what they do, and do not "
                    "repeat the helper's words."
                )
            )

    def prompt(self, label: str) -> str:
        """Generate the user's reply, resampling a corrupted completion.

        The gateway occasionally returns a completion that decodes to a wall of
        asterisks and broken words. It is a valid string, so the output retry
        does not see it; the eval would then feed it to the workflow as the
        user's turn. Resample from the same history instead.
        """
        before = list(self._conversation.messages)
        for _ in range(3):
            turn = self._conversation.send(None)
            text = turn.output.strip()
            if text and not _looks_garbled(text):
                return text
            self._conversation.messages = list(before)
        raise RuntimeError(f"User bot returned an unusable reply for {label!r}")


class JudgeVerdict(BaseModel):
    """A single validation rule's verdict on the conversation."""

    rule: str = Field(..., description="Short label of the validation rule")
    passed: bool = Field(..., description="Whether the conversation satisfies this rule")
    explanation: str | None = Field(None, description="Brief explanation if the rule was violated")
    evidence: str | None = Field(
        None,
        description="Exact quote from the transcript that supports a failure",
    )
    fix: str | None = Field(
        None,
        description="Concrete correction for a failure",
    )


class JudgeResult(BaseModel):
    """Structured evaluation of conversation quality."""

    verdicts: list[JudgeVerdict] = Field(
        ...,
        description=(
            "Individual verdicts — one per rule. The number of verdicts MUST match the number of rules provided."
        ),
    )


def _format_verdict_failure(verdict: JudgeVerdict) -> str:
    lines = [f"  [{verdict.rule}] FAIL: {verdict.explanation}"]
    if verdict.evidence:
        lines.append(f"    Evidence: {verdict.evidence}")
    if verdict.fix:
        lines.append(f"    Fix: {verdict.fix}")
    return "\n".join(lines)


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


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator of conversation quality. "
    "Evaluate the conversation transcript below against each rule. "
    "For each rule, decide PASS or FAIL and explain if it failed. "
    "The number of verdicts MUST equal the number of rules. "
    "The transcript contains only the user and assistant turns. "
    "[structured-result] marks that the workflow produced its validated "
    "result; it is not a conversation turn. Quote the exact text that "
    "supports a FAIL in 'evidence', and give the concrete correction in 'fix'."
)
JUDGE_MODEL_ENV = "CHAT_WORKFLOW_JUDGE_MODEL"


def llm_judge(
    rules: dict[str, str],
    transcript: str,
    config: Config,
) -> JudgeResult:
    """Ask an LLM to evaluate a conversation transcript against a list of rules.

    Uses a :class:`~chat_workflow.conversation.Conversation` returning a
    structured JudgeResult with a verdict for each rule. This is more reliable
    than parsing free-text YES/NO from the response.

    The judge is deliberately deterministic: temperature 0.0 and a model that
    can be pinned independently of the agent via ``CHAT_WORKFLOW_JUDGE_MODEL``.
    """
    rules_text = "\n".join(
        f"{i + 1}. {name}: {desc}" for i, (name, desc) in enumerate(rules.items())
    )
    conversation = Conversation(
        model=os.getenv(JUDGE_MODEL_ENV) or config.model,
        provider=config.provider,
        instructions=JUDGE_SYSTEM_PROMPT,
        output_type=JudgeResult,
        api_base=config.api_base,
        api_key_env=config.api_key_env,
        output_retries=config.max_retries,
        transport_retries=config.max_retries,
        timeout_seconds=config.request_timeout_seconds,
        temperature=0.0,
        extra_headers=config.extra_headers,
    )
    turn = conversation.send(
        f"Rules to evaluate:\n{rules_text}\n\nConversation transcript:\n{transcript}"
    )
    return turn.output


def _jsonable(value: Any) -> Any:
    """Convert a result object into a JSON-compatible value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def format_transcript(
    session: Session,
    result: Any = None,
    *,
    audience: str | None = None,
    given: dict[str, Any] | None = None,
) -> str:
    """Render the session tree's transcript, or the subset a person saw.

    ``audience="user"`` keeps only the turns a person saw. ``None`` keeps the
    full audit view, including thinking, tool traffic and retry prompts.
    ``given`` is the non-session context the step received; the judge needs it
    to tell "you already told me" apart from "you said nothing".
    """
    parts: list[str] = []
    if given:
        parts.append(
            "[given]\n"
            "Context the user already provided in earlier phases. It is "
            "established fact, not something the assistant invented.\n"
            + json.dumps(_jsonable(given), indent=2)
        )
    for item in session.walk():
        text = render(
            item.state.messages,
            audience="user" if audience == "user" else "all",
            result=result if item is session else item.state.final_result,
        )
        if text:
            parts.append(text)
    return "\n---\n".join(parts)


def _has_conversation(session: Session) -> bool:
    """True when the session contains at least one user or assistant turn."""
    return any(
        kind in ("user", "assistant")
        for item in session.walk()
        for kind, _ in entries(item.state.messages)
    )


def _judge_says_it_passes(verdict: JudgeVerdict) -> bool:
    """True when a failed verdict's own conclusion says there was no violation."""
    from chat_workflow.mixins import _verdict_claims_the_rule_passes

    return _verdict_claims_the_rule_passes(verdict)


@dataclass
class RunRecord:
    """One eval run, stored as JSON and rendered as text."""

    test: str
    status: str
    transcript: str
    events: list[dict[str, Any]]
    result: Any = None
    judge: dict[str, Any] | None = None
    error: str | None = None
    model: str | None = None
    provider: str | None = None
    tokens: int = 0
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "status": self.status,
            "events": self.events,
            "result": self.result,
            "judge": self.judge,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "tokens": self.tokens,
            "duration_s": self.duration_s,
            "transcript": self.transcript,
        }


def _session_events(session: Session) -> list[dict[str, Any]]:
    """Flatten the transcript of a session tree, tagging each entry."""
    events: list[dict[str, Any]] = []
    for item in session.walk():
        for kind, text in entries(item.state.messages):
            event: dict[str, Any] = {
                "type": kind,
                "content": text,
                "session": item.id,
            }
            if item.title:
                event["title"] = item.title
            events.append(event)
    return events


@contextmanager
def capture_on_failure(
    session: Session,
    label: str = "conversation",
    result: Any = None,
    judge_result: JudgeResult | None = None,
    started_at: float | None = None,
    given: dict[str, Any] | None = None,
):
    """Write the run record and a human transcript when the block fails.

    ``<label>-<ts>.json`` is the canonical record (events, result, judge
    verdicts, metadata); ``<label>-<ts>.txt`` is the rendered transcript.
    The full exception traceback remains in ``-exception.txt`` for debugging.
    """
    try:
        yield
    except BaseException as exc:
        transcript = format_transcript(session, result=result, given=given)
        env_dir = os.environ.get("CHAT_WORKFLOW_TRANSCRIPT_DIR")
        outdir = Path(env_dir) if env_dir else _DEFAULT_TRANSCRIPT_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        record = RunRecord(
            test=label,
            status="judge_failed" if judge_result is not None else "error",
            transcript=transcript,
            events=(
                _session_events(session)
            ),
            result=_jsonable(result),
            judge=judge_result.model_dump(mode="json") if judge_result else None,
            error=f"{type(exc).__name__}: {exc}",
            model=getattr(session.state, "model", None) or getattr(session.config, "model", None),
            provider=getattr(session.config, "provider", None),
            tokens=getattr(session.state.usage, "total_tokens", 0),
            duration_s=(time.time() - started_at) if started_at else None,
        )

        tx_path = outdir / f"{label}-{ts}.txt"
        tx_path.write_text(
            f"=== CONVERSATION TRANSCRIPT ({label}) ===\n"
            + transcript
            + "\n=== END TRANSCRIPT ===\n"
        )
        record_path = outdir / f"{label}-{ts}.json"
        record_path.write_text(json.dumps(record.to_dict(), indent=2, default=str) + "\n")

        import traceback

        result_section = f"\n=== RESULT ===\n{result!r}\n" if result is not None else ""
        tb_path = outdir / f"{label}-{ts}-exception.txt"
        tb_path.write_text(
            f"=== EXCEPTION DETAILS ({label}) ===\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            + result_section
            + "\n=== END EXCEPTION DETAILS ===\n"
        )

        # A BaseException (timeout, interrupt) must keep its type: unittest
        # records it as an error and continues the suite. Normal exceptions are
        # converted to a compact AssertionError with the transcript path.
        if not isinstance(exc, Exception):
            raise

        # Compact message — preserve original assertion if it's an AssertionError
        if isinstance(exc, AssertionError):
            msg = str(exc)
        else:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            msg = str(cause).split("\n")[0] if str(cause) else type(cause).__name__

        raise AssertionError(
            f"{msg}\n[Transcript: {tx_path}]\n[Record: {record_path}]\n[Details: {tb_path}]"
        ) from exc


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
        method_kwargs: Keyword args to pass to model_method. The harness creates
            the root session and wraps the step in a SessionComponent.
        user_persona: Persona prompt for the AgentIO user bot
        judge_rules: Dict mapping rule labels to descriptions.
            Defaults to DEFAULT_JUDGE_RULES.
        judge: The judge function. Callable[[dict[str, str], str, Config], JudgeResult].
            Defaults to llm_judge. Pass None to skip judging.
        config: Config instance. Defaults to make_config().
    """
    # The transcript and usage belong to the Session; no second token counter.
    config = config or make_config()
    if judge_rules is None:
        judge_rules = getattr(model_method, "__conversation_rules__", None) or DEFAULT_JUDGE_RULES

    user_bot = AgentIO(persona_prompt=user_persona, config=config)

    # Create session — state.messages records the conversation
    session = make_tools(user_bot, config=config)

    # The harness owns the root session and passes it to the step; a session
    # in method_kwargs belongs to a prebuilt harness, not to this run.
    kwargs = {
        key: value for key, value in method_kwargs.items() if key != "session"
    }
    given = {
        key: value
        for key, value in kwargs.items()
        if key != "max_turns" and value is not None
    }

    # Auto-detect unittest test method name from call stack if not provided.
    # Do this before running so failure labels are useful.
    if test_name is None:
        for frame in inspect.stack():
            locals = frame[0].f_locals
            method = locals.get("self")
            if method is not None and hasattr(method, "_testMethodName"):
                test_name = method._testMethodName
                break
        else:
            test_name = model_method.__name__

    started_at = time.time()

    def run_step(*, session: Session) -> Any:
        return model_method(**kwargs, session=session)

    # The closure hides the step's chat flag; carry it so a chat-off top-level
    # step is not greeted or closed.
    run_step._chat = getattr(model_method, "_chat", True)  # type: ignore[attr-defined]

    try:
        # The harness wraps a real workflow step in a session: greet, run, sign
        # off. A plain callable is a harness test double, not a workflow step;
        # wrapping it would need a provider and would test the fake, not the
        # harness.
        if getattr(model_method, "_is_workflow_step", False):
            result = SessionComponent().converse(run_step, session=session)
        else:
            result = run_step(session=session)
    except BaseException as exc:
        with capture_on_failure(
            session,
            label=test_name or "conversation",
            started_at=started_at,
            given=given,
        ):
            if isinstance(exc, Exception):
                raise AssertionError(f"[workflow failed] {exc}") from exc
            raise

    if result is None:
        with capture_on_failure(
            session,
            label=test_name or "conversation",
            started_at=started_at,
            given=given,
        ):
            raise AssertionError("[no typed result] workflow returned None")

    recovered = [
        text
        for item in session.walk()
        for kind, text in entries(item.state.messages)
        if kind == "validation"
    ]
    if recovered:
        line = (
            f"  [recovered] {test_name or 'conversation'}: "
            f"{len(recovered)} validation correction(s)"
        )
        print(line)
        if os.environ.get("CHAT_WORKFLOW_EVAL_REPORT"):
            report = Path("test-results/eval-report.txt")
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("a") as fh:
                fh.write(line + "\n")

    if not _has_conversation(session):
        with capture_on_failure(
            session,
            label=test_name or "conversation",
            result=result,
            started_at=started_at,
            given=given,
        ):
            raise AssertionError(
                "[no conversation] workflow returned a result without any "
                "assistant/user turns"
            )

    transcript = format_transcript(session, result=result, given=given)
    judge_transcript = format_transcript(
        session, result=result, audience="user", given=given
    )

    # The judge runs only after the workflow produced a typed result and a
    # non-empty conversation. It is the quality gate, not a substitute for the
    # structural result checks.
    failures: list[JudgeVerdict] = []
    judge_result: JudgeResult | None = None
    if judge is not None:
        judged = judge(judge_rules, judge_transcript, config)
        judge_result = judged
        failures = [
            verdict
            for verdict in judged.verdicts
            if not verdict.passed and not _judge_says_it_passes(verdict)
        ]

    # Save the human transcript on success too when env var is set
    if os.environ.get("CHAT_WORKFLOW_SAVE_TRANSCRIPT"):
        outdir = _DEFAULT_TRANSCRIPT_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        tx_path = outdir / f"success-{test_name or 'conversation'}-{int(time.time())}.txt"
        tx_path.write_text(transcript)

    if failures:
        with capture_on_failure(
            session,
            label=test_name or "conversation",
            result=result,
            judge_result=judge_result,
            started_at=started_at,
            given=given,
        ):
            raise AssertionError(
                f"[judge failed] {len(failures)}/{len(judge_rules)} rules failed:\n"
                + "\n".join(_format_verdict_failure(v) for v in failures)
            )

    return result
