import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

from .transcript import dump

__all__ = ["SessionRecord", "get_logs_dir", "log_session"]


@dataclass(frozen=True)
class SessionRecord:
    """One session's transcript and outcome, ready to store."""

    messages: Sequence[ModelMessage]
    criteria: dict[str, Any] | None
    success_judgement: bool
    feedback_text: str | None
    model: str
    turn_count: int
    context: str


@dataclass(frozen=True)
class UserFeedback:
    """The user's verdict on the session."""

    success_judgement: bool
    feedback_text: str | None


@dataclass(frozen=True)
class StoredSession:
    """The JSON document written to disk."""

    timestamp: str
    model: str
    turn_count: int
    context: str
    messages: Any
    criteria: dict[str, Any] | None
    user_feedback: UserFeedback


def get_logs_dir(home: Path | None = None) -> Path:
    logs_dir = (home or Path.home()) / ".chat-workflow" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def log_session(record: SessionRecord, *, logs_dir: Path | None = None) -> Path:
    """Write one session's transcript and outcome as JSON.

    ``messages`` is the transcript itself (PydanticAI's message list), stored
    in its own serialised form: a second, text-only message shape would lose
    thinking, tool traffic and retry prompts.
    """
    directory = logs_dir or get_logs_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = directory / f"session_{timestamp}.json"

    stored = StoredSession(
        timestamp=datetime.now().isoformat(),
        model=record.model,
        turn_count=record.turn_count,
        context=record.context,
        messages=json.loads(dump(record.messages)),
        criteria=record.criteria,
        user_feedback=UserFeedback(
            success_judgement=record.success_judgement,
            feedback_text=record.feedback_text,
        ),
    )

    log_path.write_text(json.dumps(asdict(stored), indent=2))
    return log_path
