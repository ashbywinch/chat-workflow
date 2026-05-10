"""Protocols for conversation IO, debugging, and orchestration."""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar

from .models import ConversationResult

TResult = TypeVar("TResult")


class ConversationIO(Protocol):
    def echo(self, message: str) -> None: ...

    def prompt(self, label: str) -> str: ...


class ConversationDebug(Protocol):
    """Protocol for debugging LLM conversations.

    Implement this to receive debug events during conversation flow.
    """

    def on_request(self, messages: list[dict[str, str]], model: str) -> None:
        """Called before sending request to LLM."""
        ...

    def on_response(self, response: Any, duration_ms: int) -> None:
        """Called after receiving response from LLM."""
        ...

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        ...


class ConversationOrchestratorLike(Protocol, Generic[TResult]):
    messages: list[dict[str, str]]
    turn_count: int
    model: str

    def process_turn(self, user_input: str) -> ConversationResult[TResult]: ...
