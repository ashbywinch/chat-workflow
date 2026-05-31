"""Shared helpers for eval tests that call real LLMs."""

import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel

from chat_workflow import Config, Session, SessionLog

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


def llm_judge(question: str, content: str, config: Config, max_tokens: int = 200) -> tuple[bool, str]:
    """Ask an LLM to evaluate whether *content* satisfies *question*.

    Uses the same model as the evals. The question should be a yes/no
    question about the content. Returns (passed, reasoning) where
    *passed* is True/False and *reasoning* is the judge's explanation.

    This is useful for evaluating non-deterministic outcomes like
    "is this analysis relevant to meeting minutes?" where keyword
    matching would be too brittle. The reasoning helps diagnose
    what the judge didn't like.
    """
    litellm.suppress_debug_info = True
    response = litellm.completion(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict but fair evaluator. Answer the question about the "
                    "content below. First say YES or NO on its own line, "
                    "then on the next line give a brief reason for your verdict."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContent: {content}",
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    answer = (response.choices[0].message.content or "").strip()
    lines = answer.split("\n")
    verdict = lines[0].strip().upper()
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    passed = verdict.startswith("YES")
    return passed, reasoning


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
def capture_on_failure(session: Session):
    """If the enclosed block fails, dump the full conversation transcript to stderr."""
    try:
        yield
    except Exception:
        transcript = format_transcript(session)
        print(
            "\n=== CONVERSATION TRANSCRIPT (before failure) ===\n"
            + transcript
            + "\n=== END TRANSCRIPT ===\n",
            file=sys.stderr,
        )
        raise


DEFAULT_JUDGE_PROMPT = (
    "Did the analyst propose a structure and then ask to fill in details? "
    "Avoid repeating questions? Use plain language "
    "(explaining jargon when asked is fine)? "
    "Answer YES for productive conversation, "
    "NO only if stuck in a pure questioning loop."
)


def run_multi_turn_eval(
    model_method: Callable[..., Any],
    method_kwargs: dict[str, Any],
    user_persona: str,
    judge_prompt: str | None = None,
    judge: Callable = llm_judge,
    config: Config | None = None,
) -> Any:
    """Run a multi-turn workflow eval with an LLM user bot and optional LLM judge.

    Hands off to the model method (which is an @atomic_workflow-decorated
    classmethod), then verifies turn efficiency, runs the judge on the
    conversation transcript, and dumps the transcript on failure.

    Args:
        model_method: The @atomic_workflow or @composite_workflow classmethod to test
        method_kwargs: Keyword args to pass to model_method (must include session)
        user_persona: Persona prompt for the AgentIO user bot
        judge_prompt: Prompt for the LLM judge. Defaults to DEFAULT_JUDGE_PROMPT.
        judge: The judge function (defaults to llm_judge). Pass None to skip judging.
        config: Config instance. Defaults to make_config().
    """
    config = config or make_config()
    judge_prompt = judge_prompt or DEFAULT_JUDGE_PROMPT

    user_bot = AgentIO(persona_prompt=user_persona, config=config)

    # Create session — state.messages records the conversation
    session = make_tools(user_bot)

    # Inject session into method_kwargs if not present
    kwargs = dict(method_kwargs)
    kwargs.setdefault("session", session)

    with capture_on_failure(session):
        result = model_method(**kwargs)

        # Turn efficiency assertion
        max_turns = kwargs.get("max_turns", 10)
        assert session.state.turn_count < max_turns, (
            f"Workflow burned all {session.state.turn_count} turns — likely stuck in questioning loop. "
            f"User bot had to provide {len(user_bot.outputs)} responses."
        )

        # LLM judge on the full conversation
        if judge is not None:
            transcript = format_transcript(session)
            ok, reason = judge(judge_prompt, transcript, config)
            assert ok, (
                f"Conversation quality issue detected.\n"
                f"Judge's reasoning: {reason}\n"
                f"Transcript:\n{transcript}"
            )

    return result


def run_one_shot_eval(
    response_model: type[BaseModel],
    system_prompt: str,
    initial_message: str,
    user_turn: str,
    config: Config | None = None,
) -> Any:
    """Run a one-shot workflow eval — single LLM call with fixed prompts.

    Constructs an AtomicWorkflow with standard boilerplate, processes one
    turn, and returns the result for structural assertions.
    """
    from chat_workflow import (
        AgentResponse,
        AtomicWorkflow,
        AtomicWorkflowConfig,
        AtomicWorkflowFailedError,
        TurnResult,
    )

    config = config or make_config()

    orchestrator = AtomicWorkflow(
        config=AtomicWorkflowConfig(
            system_prompt=system_prompt,
            response_model=AgentResponse[response_model],
            max_turns=3,
            model=config.model,
            provider=config.provider,
            max_retries=config.max_retries,
            request_timeout_seconds=config.request_timeout_seconds,
            initial_messages=[{"role": "user", "content": initial_message}],
            on_continue=lambda action: TurnResult[response_model].continuing(action.message or ""),
            on_success=lambda action: TurnResult[response_model].success(action.result),
            on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
        )
    )

    result = orchestrator.process_turn(user_turn)
    return result.result if result else None


def make_meeting_analysis():
    """Create a standard meeting-minutes ProcessAnalysis for eval tests."""
    from workflows.workflow.models import ProcessAnalysis

    return ProcessAnalysis(
        phases=["Note-taking", "Review & Clarify", "Draft Minutes", "Review & Approve"],
        activities=[
            "Take meeting notes", "Review notes for clarity",
            "Identify action items", "Write minutes draft",
            "Circulate for review", "Incorporate feedback",
            "Distribute final minutes",
        ],
        orchestrating_component="Meeting Organizer",
        participants=["Meeting Attendees", "Note Taker", "Reviewers"],
    )
