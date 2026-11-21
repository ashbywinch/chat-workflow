from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from .atomic_workflow import AtomicWorkflow


@dataclass(frozen=True)
class TranscriptSegment:
    """One atomic step's slice of the session transcript.

    A session's messages are the concatenation of several steps' transcripts.
    ``chat`` says whether the step had a user of its own: a chat-off step's
    turns are internal and never appeared in the conversation.
    """

    start: int
    end: int
    chat: bool = True


@dataclass
class SessionLog:
    """Accumulates session state across the entire workflow run.

    ``messages`` is the transcript — PydanticAI's message list, thinking, tool
    traffic and retry prompts included — not a hand-built rendering of it.
    ``segments`` marks which slice of it belongs to which step, so the user view
    can be projected per step instead of treating the concatenation as one
    conversation. ``usage`` is the run's token and request total across every
    step.
    """

    messages: list[ModelMessage] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    model: str = "unknown"  # TODO: this is not type safe
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None


def _record_atomic_workflow[T](
    log: SessionLog,
    workflow: AtomicWorkflow[T],
) -> None:
    """Fold an atomic workflow's new transcript and usage into the session log.

    A step may be handed the session's transcript as its starting context. Only
    the messages the step added are folded back, so the history is not
    duplicated. The slice is recorded as a segment so the projection can find
    each step's own final answer instead of only the run's last one.
    """
    initial = getattr(workflow, "initial_message_count", 0)
    start = len(log.messages)
    log.messages.extend(workflow.messages[initial:])
    log.segments.append(
        TranscriptSegment(
            start=start,
            end=len(log.messages),
            chat=getattr(workflow, "chat", True),
        )
    )
    log.usage.incr(workflow.usage)
    log.turn_count += workflow.turn_count
    log.model = workflow.model
