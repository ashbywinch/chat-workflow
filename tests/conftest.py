#!/usr/bin/env python3
"""
Common test configuration and fixtures.
"""

import os
import signal
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any

from chat_workflow import AgentResponse, AtomicWorkflowConfig, AtomicWorkflowFailedError, TurnResult
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
    Also tracks LLM token usage and test timing for the cost report.

    Usage:
        @timeout(10)
        def test_something(self):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Reset token counter before each test so we capture per-test usage
            _reset_token_counter()
            start = time.time()

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
                # Write token + timing data to eval report
                _write_eval_report(func, start)

        return wrapper

    return decorator


def _reset_token_counter() -> None:
    try:
        from tests.evals.helpers import _reset_token_counter as _rtc
        _rtc()
    except ImportError:
        pass


def _write_eval_report(func, start: float) -> None:
    """Append timing and token data to test-results/eval-report.txt.

    Only writes if CHAT_WORKFLOW_EVAL_REPORT is set.  Uses the unittest
    method name if available (``self._testMethodName``) or the function
    name as fallback.
    """
    if not os.environ.get("CHAT_WORKFLOW_EVAL_REPORT"):
        return
    try:
        from tests.evals.helpers import _read_token_counter
        tokens = _read_token_counter()
        duration = time.time() - start

        # Try to get the test method name from unittest
        import inspect
        method_name = func.__name__
        for frame in inspect.stack():
            locals = frame[0].f_locals
            method = locals.get("self")
            if method is not None and hasattr(method, "_testMethodName"):
                method_name = method._testMethodName
                break

        report_dir = Path(__file__).parent.parent / "test-results"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "eval-report.txt"
        with open(report_path, "a") as f:
            f.write(f"  [{method_name}] {duration:.0f}s  {tokens} tok\n")
    except Exception:
        pass  # Don't let tracking failures break tests


class FakeConfig:
    """Lightweight config double for tests. Production uses Config(Path(...))."""

    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False
    model_supports_tools = False
    api_base = None
    api_key_env = None


def make_valid_criteria() -> EvaluationCriteria:
    """Create a standard valid EvaluationCriteria for tests."""
    return EvaluationCriteria(
        context="test context",
        criteria=[
            Criterion(name="budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ],
    )


def make_atomic_workflow_config(
    response_model_override: type | None = None,
    max_turns: int = 10,
    **overrides: Any,
) -> AtomicWorkflowConfig:
    """Create a standard AtomicWorkflowConfig for testing."""
    kwargs = dict(
        system_prompt="Test prompt",
        response_model=response_model_override or AgentResponse[EvaluationCriteria],
        max_turns=max_turns,
        on_continue=lambda action: TurnResult[EvaluationCriteria].continuing(action.message or ""),
        on_success=lambda action: TurnResult[EvaluationCriteria].success(action.result),
        on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
    )
    kwargs.update(overrides)
    return AtomicWorkflowConfig(**kwargs)


class MockInstructorClient:
    """Mock instructor client for testing LLM interactions."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_call_args = None
        self.last_call_kwargs = None
        self.chat = self.MockChatCompletions(self)

    class MockChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self

        def create(self, model=None, messages=None, response_model=None, max_retries=None, **kwargs):
            self.parent.call_count += 1
            self.parent.last_call_args = (model, messages, response_model, max_retries)
            self.parent.last_call_kwargs = kwargs
            if self.parent.responses:
                response = self.parent.responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
            return AgentResponse[EvaluationCriteria](intent="continue", message="Test question")

    def get_client(self, provider=None):
        return self
