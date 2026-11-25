"""Debug utilities for LLM conversation tracing."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, ClassVar

from pydantic_ai.messages import ModelMessage

from .transcript import TranscriptEntry
from .transcript import entries as transcript_entries
from .transcript import render as render_transcript


class StreamingDebug:
    """A debug callback that prints LLM interactions to stdout in real-time.

    Usage:
        debug = StreamingDebug()
        workflow = AtomicWorkflow(..., debug=debug)
    """

    # ``%f`` prints microseconds; three digits of them are milliseconds.
    _MILLISECOND_DIGITS: ClassVar[int] = 3

    def __init__(self, file: Any = None, include_timestamps: bool = True):
        self.file = file or sys.stderr
        self.include_timestamps = include_timestamps
        self._request_start: datetime | None = None
        self._shown_entries = 0
        self._last_messages: list[ModelMessage] = []

    def _timestamp(self) -> str:
        if self.include_timestamps:
            now = datetime.now().strftime("%H:%M:%S.%f")
            return f"[{now[: -self._MILLISECOND_DIGITS]}] "
        return ""

    def _print(self, message: str) -> None:
        print(message, file=self.file, flush=True)

    def _print_entries(self, entries: list[TranscriptEntry], start: int, truncate: bool = True) -> None:
        for kind, text in entries[start:]:
            if truncate and len(text) > 500:
                text = text[:500] + "..."
            self._print(f"{self._timestamp()}{kind}: {text}")

    def on_request(self, messages: list[ModelMessage], model: str) -> None:
        self._request_start = datetime.now()
        self._last_messages = messages
        entries = transcript_entries(messages)
        self._print(f"{self._timestamp()}━━━ LLM REQUEST ━━━")
        self._print(f"{self._timestamp()}Model: {model}")
        if self._shown_entries > 0 and self._shown_entries < len(entries):
            self._print(f"{self._timestamp()}  [... {self._shown_entries} prior entries]")
        self._print_entries(entries, start=self._shown_entries, truncate=True)
        self._shown_entries = len(entries)
        self._print(f"{self._timestamp()}Waiting for response...")

    def on_response(self, response: Any, duration_ms: float) -> None:
        self._print(f"{self._timestamp()}━━━ LLM RESPONSE ({duration_ms:.0f}ms) ━━━")
        try:
            if hasattr(response, "model_dump"):
                self._print(f"{self._timestamp()}{json.dumps(response.model_dump(), indent=2, default=str)}")
            else:
                self._print(f"{self._timestamp()}{response}")
        # lucidlint: ignore swallow debug rendering is a boundary: an unrenderable response still prints its repr
        except Exception:
            self._print(f"{self._timestamp()}{response}")

    def on_error(self, error: Exception) -> None:
        lines: list[str] = []
        if self._last_messages:
            lines.append(f"{self._timestamp()}━━━ CONVERSATION TRANSCRIPT ━━━")
            lines.append(render_transcript(self._last_messages))
        lines.append(f"{self._timestamp()}━━━ ERROR ━━━")
        lines.append(f"{self._timestamp()}{type(error).__name__}: {error}")
        self._print("\n".join(lines))
