from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from .atomic_workflow import AtomicWorkflow


@dataclass
class SessionLog:
    """Accumulates session state across the entire workflow run.

    ``messages`` is the transcript — PydanticAI's message list, thinking, tool
    traffic and retry prompts included — not a hand-built rendering of it.
    ``usage`` is the run's token and request total across every step.
    """

    messages: list[ModelMessage] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    model: str = "unknown"  # TODO: this is not type safe
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None


def _record_atomic_workflow[T](
    log: SessionLog,
    workflow: AtomicWorkflow[T],
) -> None:
    """Fold an atomic workflow's transcript and usage into the session log."""
    log.messages.extend(workflow.messages)
    log.usage.incr(workflow.usage)
    log.turn_count += workflow.turn_count
    log.model = workflow.model
