#!/usr/bin/env python3
"""
Common test configuration and fixtures.
"""

import signal
import sys
from functools import wraps
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from chat_workflow import (
    AgentAnswer,
    AtomicWorkflowConfig,
    AtomicWorkflowFailedError,
    Conversation,
    FinalAnswer,
    TurnResult,
)
from workflows.evaluation_criteria import Criterion, EvaluationCriteria

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TimeoutError(Exception):
    """Raised when a test exceeds its timeout."""

    pass


def timeout(seconds: int):
    """
    Decorator that raises TimeoutError if the wrapped function exceeds the timeout.

    Usage:
        @timeout(10)
        def test_something(self):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")

            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                return func(*args, **kwargs)
            finally:
                # Restore original handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


class FakeConfig:
    """Lightweight config double for tests. Production uses Config(Path(...))."""

    model = "test-model"
    provider = "test-provider"
    api_base = None
    api_key_env = None
    max_retries = 3
    request_timeout_seconds = 30
    debug = False
    extra_headers: dict[str, str] = {}
    temperature: float | None = None
    model_client: Any = None


def make_valid_criteria() -> EvaluationCriteria:
    """Create a standard valid EvaluationCriteria for tests."""
    return EvaluationCriteria(
        context="test context",
        criteria=[
            Criterion(name="budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ],
    )


def _success(action) -> TurnResult:
    """The success callback: a success turn always carries a result."""
    if action.result is None:
        raise AtomicWorkflowFailedError("SUCCESS intent returned no result")
    return TurnResult[EvaluationCriteria].success(
        action.result, message=action.message
    )


def make_atomic_workflow_config(
    response_model_override: type | None = None,
    max_turns: int | None = 10,
    **overrides: Any,
) -> AtomicWorkflowConfig:
    """Create a standard AtomicWorkflowConfig for testing."""
    kwargs = dict(
        system_prompt="Test prompt",
        response_model=response_model_override or AgentAnswer[EvaluationCriteria],
        max_turns=max_turns,
        on_continue=lambda action: TurnResult[EvaluationCriteria].continuing(action.message or ""),
        on_success=_success,
        on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
    )
    kwargs.update(overrides)
    return AtomicWorkflowConfig(**kwargs)


def text_response(payload: Any) -> ModelResponse:
    """A model response that calls the ``answer`` tool with *payload*.

    The answer travels as a tool call: the endpoints behind this model enforce
    a tool's schema and ignore ``response_format``. The schema wraps the
    intents' union in an object under ``response``, so the arguments do too.
    """
    response = payload
    if isinstance(response, dict) and response.get("intent") == "success":
        # A success is a turn: it says what it did. Scripted tests focus on
        # the result, so the helper supplies the words; the protocol test
        # below covers the rejection of a silent success.
        response.setdefault("message", "Done.")
    return ModelResponse(parts=[ToolCallPart("answer", payload)])


class ScriptedModel:
    """A FunctionModel body that replays scripted responses and counts calls.

    An item may be a payload (serialised as the response text), a
    ``ModelResponse`` (for blank or thinking-only responses), or a callable
    returning either. The last item repeats once the script runs out.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: Any) -> ModelResponse:
        del info
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        item = self.responses[index]
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ModelResponse):
            return item
        return text_response(item)


def make_conversation(
    responses: list[Any],
    *,
    output_type: type | None = None,
    final_output_type: type | None = None,
    **overrides: Any,
) -> Any:
    """A Conversation on a scripted model: no network, no provider."""
    script = ScriptedModel(responses)
    conversation = Conversation(
        model="scripted-model",
        instructions="Test instructions",
        output_type=output_type or AgentAnswer[EvaluationCriteria],
        final_output_type=final_output_type or FinalAnswer[EvaluationCriteria],
        model_client=FunctionModel(script),
        **overrides,
    )
    conversation.script = script  # type: ignore[attr-defined]  # tests read the calls
    return conversation
