from dataclasses import dataclass, field
from typing import Any

from chat_workflow.protocols import ConversationOrchestratorLike


@dataclass
class ConversationLog:
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
    self,
    orchestrator: ConversationOrchestratorLike[T],
) -> None:
    """Record orchestrator state into the conversation flow state.

    Appends the orchestrator's messages, increments the turn count,
    and records the model name.

    Args:
        state: The conversation flow state to update.
        orchestrator: The orchestrator whose state to record.
    """
    self.messages.extend(orchestrator.messages)
    self.turn_count += orchestrator.turn_count
    self.model = orchestrator.model
