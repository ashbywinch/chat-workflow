import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

from .transcript import dump


def get_logs_dir() -> Path:
    logs_dir = Path.home() / ".chat-workflow" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def log_session(
    messages: Sequence[ModelMessage],
    criteria: dict[str, Any] | None,
    success_judgement: bool,
    feedback_text: str | None,
    model: str,
    turn_count: int,
    context: str,
) -> Path:
    """Write one session's transcript and outcome as JSON.

    ``messages`` is the transcript itself (PydanticAI's message list), stored
    in its own serialised form: a second, text-only message shape would lose
    thinking, tool traffic and retry prompts.
    """
    logs_dir = get_logs_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"session_{timestamp}.json"
    log_path = logs_dir / log_filename

    session_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "turn_count": turn_count,
        "context": context,
        "messages": json.loads(dump(messages)),
        "criteria": criteria,
        "user_feedback": {
            "success_judgement": success_judgement,
            "feedback_text": feedback_text,
        },
    }

    log_path.write_text(json.dumps(session_data, indent=2))
    return log_path
