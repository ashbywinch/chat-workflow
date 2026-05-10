#!/usr/bin/env python3
"""
Common test configuration and fixtures.
"""

import signal
import sys
from functools import wraps
from pathlib import Path
from typing import Any

from chat_workflow import ConversationAction, ConversationResult
from chat_workflow.exceptions import ConversationFailedError
from chat_workflow.orchestrator_config import OrchestratorConfig
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
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


def make_valid_criteria() -> EvaluationCriteria:
    """Create a standard valid EvaluationCriteria for tests."""
    return EvaluationCriteria(
        context="test context",
        criteria=[
            Criterion(name="budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ],
    )


def make_orchestrator_config(
    response_model_override: type | None = None,
    max_turns: int = 10,
    **overrides: Any,
) -> OrchestratorConfig:
    """Create a standard OrchestratorConfig for testing."""
    kwargs = dict(
        system_prompt="Test prompt",
        response_model=response_model_override or ConversationAction[EvaluationCriteria],
        max_turns=max_turns,
        on_continue=lambda action: ConversationResult[EvaluationCriteria].continuing(action.message or ""),
        on_success=lambda action: ConversationResult[EvaluationCriteria].success(action.result),
        on_failure=lambda action: ConversationFailedError(action.message or "No reason given"),
    )
    kwargs.update(overrides)
    return OrchestratorConfig(**kwargs)


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
            return ConversationAction[EvaluationCriteria](action="continue", message="Test question")

    def get_client(self, provider=None):
        return self
