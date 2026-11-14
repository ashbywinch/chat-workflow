from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .atomic_workflow_step import AtomicWorkflowStep

TranscriptEventType = Literal[
    "user",
    "assistant",
    "framework",
    "validation",
    "result",
    "failure",
]


@dataclass
class TranscriptEvent:
    """One typed entry in the conversation audit log.

    The event log is the canonical transcript. ``messages`` remains the LLM
    API history and must stay ``role``/``content``; the events add the type
    that a text transcript loses: who said it, whether the harness produced it,
    which turn it belongs to, and any structured metadata.
    """

    type: TranscriptEventType
    content: Any = None
    turn: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        event: dict[str, Any] = {"type": self.type, "content": self.content}
        if self.turn is not None:
            event["turn"] = self.turn
        if self.meta:
            event["meta"] = self.meta
        return event


@dataclass
class SessionLog:
    """Accumulates session state across the entire workflow run.

    Tracks the typed event log, the LLM message history, the model used, turn
    count, and the initial/final results produced across all atomic workflow
    steps.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    # Canonical transcript. ``messages`` is the API history; this is the audit
    # log the eval harness records, renders, and judges.
    events: list[TranscriptEvent] = field(default_factory=list)
    model: str = "unknown"  # TODO: this is not type safe
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None
    # Validation failures that the framework fed back, including whether the
    # workflow recovered on that turn. Recovered events are near-misses: they
    # show which rules the model can fix with a good error message, and which
    # prompts are still weak.
    validation_events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        event_type: TranscriptEventType,
        content: Any = None,
        *,
        turn: int | None = None,
        **meta: Any,
    ) -> TranscriptEvent:
        """Append a typed event to the canonical transcript."""
        event = TranscriptEvent(type=event_type, content=content, turn=turn, meta=meta)
        self.events.append(event)
        return event


def _record_atomic_workflow_step[T](
    log: SessionLog,
    workflow: AtomicWorkflowStep[T],
) -> None:
    """Record an atomic workflow step's state into the session log.

    Appends the step's typed events and messages, increments the turn count,
    records the model name, and folds validation events into the audit log.
    """
    log.events.extend(getattr(workflow, "events", []))
    log.messages.extend(workflow.messages)
    log.turn_count += workflow.turn_count
    log.model = workflow.model
    new_validation_events = getattr(workflow, "validation_events", [])
    log.validation_events.extend(new_validation_events)
    for event in new_validation_events:
        log.events.append(
            TranscriptEvent(
                type="validation",
                content=event.get("reason"),
                turn=event.get("turn"),
                meta={
                    "attempt": event.get("attempt"),
                    "recovered": bool(event.get("recovered")),
                },
            )
        )
