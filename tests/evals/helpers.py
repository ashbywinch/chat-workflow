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
    """Load the eval config and prove it is pointed at Cloudflare AI Gateway.

    All house LLM traffic goes through the gateway (`skill://cloudflare-ai-gateway`):
    the gateway owns the provider keys and fails over between them. A preset
    with the `openai` provider and no explicit `api_base` relies on
    `OPENAI_BASE_URL`; without it litellm would silently call api.openai.com.
    """
    config = Config(_CONFIG_PATH)
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if config.provider == "openai" and (not base_url or "{" in base_url):
        raise RuntimeError(
            "OPENAI_BASE_URL is not set (or still contains the .env.example "
            "placeholder). Evals must route through Cloudflare AI Gateway — run "
            "`make setup` to create .env from .env.example, fill in the account "
            "and gateway, then set OPENAI_API_KEY (the gateway token)."
        )
    return config


def make_tools(io, config: Config | None = None) -> Session:
    return Session(
        io=io,
        state=SessionLog(),
        config=config or make_config(),
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
        """Generate a user response using the LLM.

        Retries an empty completion instead of returning it: an empty user turn
        would consume one of the agent's turns and make a healthy conversation
        look like a loop.
        """
        # Build messages: system persona + conversation history
        messages = [{"role": "system", "content": self.persona_prompt}]
        messages.extend(self._history)

        # Suppress litellm debug output
        litellm.suppress_debug_info = True

        last_finish: str | None = None
        last_content = ""
        last_reasoning = ""
        for attempt in range(3):
            try:
                response = litellm.completion(
                    model=self.config.model,
                    messages=messages,
                    # A reasoning model spends part of its budget on hidden
                    # reasoning; leave enough room for the user's reply after it.
                    max_tokens=8000,
                    temperature=0.7,
                    # The gateway cache must be bypassed on every call, including
                    # the user bot. A cached user reply replays an earlier
                    # conversation and makes a green run meaningless.
                    extra_headers=getattr(self.config, "extra_headers", None),
                )
            except Exception as exc:
                # A 5xx or a malformed gateway response says nothing about the
                # conversation. Resend; do not let infrastructure look like a
                # failed user turn.
                from chat_workflow.atomic_workflow_step import (
                    _is_transient_provider_error,
                )

                if attempt == 2 or not _is_transient_provider_error(exc):
                    raise
                time.sleep(2**attempt)
                continue

            choice = response.choices[0]
            last_finish = getattr(choice, "finish_reason", None)
            last_content = (choice.message.content or "").strip()
            last_reasoning = getattr(choice.message, "reasoning_content", None) or ""
            if last_content:
                self._history.append({"role": "assistant", "content": last_content})
                return last_content
            # A max_tokens cutoff is not something the model can fix by being
            # asked again; retrying the same request only burns tokens.
            if last_finish == "length":
                raise RuntimeError(
                    f"User bot hit max_tokens before replying for {label!r} "
                    f"(model={self.config.model}, finish_reason='length', "
                    f"reasoning={last_reasoning[:200]!r}). Increase max_tokens "
                    "or shorten the conversation history."
                )

        raise RuntimeError(
            f"User bot returned an empty response three times for {label!r} "
            f"(model={self.config.model}, finish_reason={last_finish!r}, "
            f"content={last_content!r}, "
            f"reasoning={last_reasoning[:200]!r})"
        )


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

    Uses Instructor (via get_client) to return a structured JudgeResult
    with a verdict for each rule. This is more reliable than parsing
    free-text YES/NO from the response.

    The judge is deliberately deterministic: temperature 0.0 and a model that
    can be pinned independently of the agent via ``CHAT_WORKFLOW_JUDGE_MODEL``.
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
        model=os.getenv(JUDGE_MODEL_ENV) or config.model,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Rules to evaluate:\n{rules_text}\n\nConversation transcript:\n{transcript}",
            },
        ],
        response_model=JudgeResult,
        max_retries=config.max_retries,
        timeout=config.request_timeout_seconds,
        temperature=0.0,
        extra_headers=getattr(config, "extra_headers", None),
    )
    return result


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
    """Render the canonical event log as text.

    ``audience="user"`` keeps only the entries the user saw: their turns, the
    assistant's replies, and anything the harness echoed to the screen.
    Corrections, validation feedback and the result JSON are internal, so they
    stay out — a proofreader must not see text the user never saw. ``None``
    keeps everything, for the audit view.

    The audience is recorded on each event, so the projection is a property of
    the data rather than of who is asking for it.

    ``given`` is the non-session context the method already received (the
    inputs from earlier phases). The judge needs it to tell "you already told
    me" apart from "you said nothing"; it is rendered before the turns.

    Sessions created before the event log (or tests that append directly to
    ``messages``) fall back to the message-based rendering.
    """
    parts: list[str] = []
    if given:
        parts.append(
            "[given]\n"
            "Context the user already provided in earlier phases. It is "
            "established fact, not something the assistant invented.\n"
            + json.dumps(_jsonable(given), indent=2)
        )
    validation: list[str] = []
    has_result = False
    events = getattr(session.state, "events", None) or []

    if events:
        for event in events:
            if event.type == "result":
                has_result = True
                continue
            if audience is not None and event.audience != audience:
                continue
            if event.type == "validation":
                status = "recovered" if event.meta.get("recovered") else "unrecovered"
                validation.append(f"[{status}] turn {event.turn}: {str(event.content or '')[:800]}")
                continue
            parts.append(f"[{event.type}]\n{event.content}")
    else:
        for message in session.state.messages:
            role = message.get("role", "?")
            content = message.get("content", "")
            if role == "system":
                continue
            stripped = content.lstrip()
            if stripped.startswith("[framework]"):
                if audience is None:
                    parts.append(f"[framework]\n{content}")
            elif role == "assistant" and stripped.startswith("{") and '"intent"' in stripped:
                # A rejected completion envelope; never conversation.
                if audience is None:
                    parts.append(f"[framework]\n{content}")
            else:
                parts.append(f"[{role}]\n{content}")

    if validation:
        parts.append("[validation events]\n" + "\n".join(validation))

    has_result = has_result or result is not None
    if has_result:
        if audience is not None:
            parts.append("[structured-result]\n(workflow produced its validated result)")
        else:
            rendered = (
                result.model_dump_json(indent=2)
                if isinstance(result, BaseModel)
                else json.dumps(_jsonable(result), indent=2)
                if result is not None
                else "(workflow produced its validated result)"
            )
            parts.append(f"[structured-result]\n{rendered}")

    return "\n---\n".join(parts)


def _has_conversation(session: Session) -> bool:
    """True when the session contains at least one user or assistant turn."""
    events = getattr(session.state, "events", None) or []
    if any(event.type in ("user", "assistant") for event in events):
        return True
    return any(
        message.get("role") in ("user", "assistant")
        for message in session.state.messages
    )


def _judge_says_it_passes(verdict: JudgeVerdict) -> bool:
    """True when a failed verdict's own conclusion says there was no violation.

    The judge occasionally sets ``passed=False`` while its explanation ends
    "No violation found" or "does not violate the rule". That is a malformed
    verdict, not a failure; discard it exactly as the object-rule judge does.
    """
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


def _messages_as_events(session: Session) -> list[dict[str, Any]]:
    """Fallback event list for sessions that only have ``messages``."""
    events: list[dict[str, Any]] = []
    for message in session.state.messages:
        role = message.get("role", "?")
        if role == "system":
            continue
        content = message.get("content", "")
        stripped = content.lstrip()
        event_type = role
        if stripped.startswith("[framework]") or (
            role == "assistant" and stripped.startswith("{") and '"intent"' in stripped
        ):
            event_type = "framework"
        events.append({"type": event_type, "content": content})
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
                [event.to_dict() for event in session.state.events]
                if getattr(session.state, "events", None)
                else _messages_as_events(session)
            ),
            result=_jsonable(result),
            judge=judge_result.model_dump(mode="json") if judge_result else None,
            error=f"{type(exc).__name__}: {exc}",
            model=getattr(session.state, "model", None) or getattr(session.config, "model", None),
            provider=getattr(session.config, "provider", None),
            tokens=_read_token_counter(),
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

    Hands off to the model method (which is an @atomic_workflow_step-decorated
    classmethod), then verifies turn efficiency, runs the judge on the
    conversation transcript, and dumps the transcript on failure.

    Args:
        model_method: The @atomic_workflow_step or @composite_workflow_step classmethod to test
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
    session = make_tools(user_bot, config=config)

    # Inject session into method_kwargs if not present
    kwargs = dict(method_kwargs)
    kwargs.setdefault("session", session)
    given = {
        key: value
        for key, value in kwargs.items()
        if key not in ("session", "max_turns") and value is not None
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
    try:
        result = model_method(**kwargs)
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
        event
        for event in session.state.events
        if event.type == "validation" and event.meta.get("recovered")
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
