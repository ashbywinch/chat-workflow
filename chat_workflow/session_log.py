from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .atomic_workflow import AtomicWorkflow


@dataclass
class SessionLog:
    """Accumulates session state across the entire workflow run.

    Tracks messages exchanged, the model used, turn count, and
    the initial/final results produced across all atomic workflows.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    model: str = "unknown"  # TODO: this is not type safe
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None


def _record_atomic_workflow[T](
    log: SessionLog,
    workflow: AtomicWorkflow[T],
) -> None:
    """Record an atomic workflow's state into the session log.

    Appends the workflow's messages, increments the turn count,
    and records the model name.
    """
    log.messages.extend(workflow.messages)
    log.turn_count += workflow.turn_count
    log.model = workflow.model