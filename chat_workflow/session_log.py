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

# Whether the user saw this entry. The audit log holds everything; a
# conversation transcript is the user-facing subset. This is a property of the
# event, so every consumer filters on the data rather than passing a flag that
# names itself.
Audience = Literal["user", "internal"]

# Why a rejected response was rejected. These also appear as event kinds, so
# the same name describes the failure and its correction.
# A returned value failed validation. Nothing else is corrected: the message
# is not checked at runtime, only the value the step returns.
CorrectionKind = Literal["validation_correction"]

# Harness activity that has no user-facing content of its own.
HarnessEventKind = Literal[
    "composite_echo",
    "composite_prompt",
    "rejected_response",
    "provider_retry",
    "repaired_escape",
]

TranscriptEventKind = CorrectionKind | HarnessEventKind


@dataclass
class TranscriptEvent:
    """One typed entry in the conversation audit log.

    The event log is the canonical transcript. ``messages`` remains the LLM
    API history and must stay ``role``/``content``; the events add what a text
    transcript loses: who said it, whether the user saw it, which turn it
    belongs to, and what kind of entry it is.
    """

    type: TranscriptEventType
    content: Any = None
    turn: int | None = None
    # The specific entry within its type: ``composite_echo``,
    # ``repetition_correction``, ``provider_retry``, ``rejected_response``...
    kind: TranscriptEventKind | None = None
    # None derives it from the type: a user or assistant turn is what the
    # user saw, everything else is internal unless it opts in.
    audience: Audience | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.audience is None:
            self.audience = "user" if self.type in ("user", "assistant") else "internal"

    def to_dict(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": self.type,
            "content": self.content,
            "audience": self.audience,
        }
        if self.turn is not None:
            event["turn"] = self.turn
        if self.kind is not None:
            event["kind"] = self.kind
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

    def record(
        self,
        event_type: TranscriptEventType,
        content: Any = None,
        *,
        turn: int | None = None,
        kind: TranscriptEventKind | None = None,
        audience: Audience | None = None,
        **meta: Any,
    ) -> TranscriptEvent:
        """Append a typed event to the canonical transcript."""
        event = TranscriptEvent(
            type=event_type,
            content=content,
            turn=turn,
            kind=kind,
            audience=audience,
            meta=meta,
        )
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
    for event in getattr(workflow, "validation_events", []):
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
