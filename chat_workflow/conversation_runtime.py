"""Workflow-agnostic runtime helpers for interactive conversation flows."""
from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import ConversationResult
from .orchestrator import StructuredConversationOrchestrator  # noqa: F401  # re-export for backwards compat
from .protocols import ConversationIO, ConversationOrchestratorLike

TResult = TypeVar("TResult")


@dataclass
class ConversationFlowState:
    """Holds the accumulated state of a conversation flow.

    Tracks messages exchanged, the model used, turn count, and
    the initial/final results produced by the orchestrator.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    model: str = "unknown"  # TODO: this is not type safe
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None


def _record_orchestrator[T](
    state: ConversationFlowState,
    orchestrator: ConversationOrchestratorLike[T],
) -> None:
    """Record orchestrator state into the conversation flow state.

    Appends the orchestrator's messages, increments the turn count,
    and records the model name.

    Args:
        state: The conversation flow state to update.
        orchestrator: The orchestrator whose state to record.
    """
    state.messages.extend(orchestrator.messages)
    state.turn_count += orchestrator.turn_count
    state.model = orchestrator.model


@dataclass
class ConversationTools:
    """Provides chat interaction with IO and state tracking.

    Wraps an :class:`ConversationIO` instance and a
    :class:`ConversationFlowState` to offer a simple ``chat`` method
    that drives a multi-turn conversation with the user.
    """

    io: ConversationIO
    state: ConversationFlowState
    config: Any = None

    def chat[TResult](
        self,
        orchestrator: ConversationOrchestratorLike[TResult],
        first_user_input: str,
    ) -> ConversationResult[TResult]:
        """Run a multi-turn conversation to completion.

        Sends the first user input, then loops over user prompts
        until the orchestrator signals completion.

        Args:
            orchestrator: The orchestrator driving the conversation.
            first_user_input: The initial user message to process.

        Returns:
            The final :class:`ConversationResult` produced by the
            orchestrator.
        """
        try:
            result = orchestrator.process_turn(first_user_input)
            self.io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                user_input = self.io.prompt("\nYou")
                result = orchestrator.process_turn(user_input)
                self.io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_orchestrator(self.state, orchestrator)


def _get_return_type(func: Callable[..., Any]) -> type[BaseModel] | None:
    """Extract the return type annotation from a function.

    Uses ``typing.get_type_hints`` to resolve the ``return``
    annotation of *func*.

    Args:
        func: The function whose return type to extract.

    Returns:
        The resolved return type, or ``None`` if no return annotation
        is present.
    """
    hints = typing.get_type_hints(func)
    return_type = hints.get("return")

    if return_type is None:
        return None

    return return_type



