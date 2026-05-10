"""Debug utilities for LLM conversation tracing."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Generic, TypeVar

from .models import ConversationAction
from .protocols import ConversationDebug

TResult = TypeVar("TResult")


class _DebugTimer(Generic[TResult]):
    """Context manager that times an LLM call and emits debug events.

    Usage:
        timer = _DebugTimer(debug, messages, model)
        with timer:
            response = client.chat.completions.create(...)
        timer.emit_response(response)
    """

    def __init__(
        self,
        debug: ConversationDebug | None,
        messages: list[dict[str, str]],
        model: str,
    ):
        self._debug = debug
        self._messages = messages
        self._model = model
        self._start: datetime | None = None

    def __enter__(self) -> _DebugTimer[TResult]:
        if self._debug:
            self._debug.on_request(self._messages, self._model)
            self._start = datetime.now()
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def emit_response(self, response: ConversationAction[TResult]) -> None:
        if self._debug and self._start is not None:
            delta = datetime.now() - self._start
            duration_ms = delta.seconds * 1000 + delta.microseconds // 1000
            self._debug.on_response(response, duration_ms)


class StreamingDebug:
    """A debug callback that prints LLM interactions to stdout in real-time.

    Usage:
        debug = StreamingDebug()
        orchestrator = StructuredConversationOrchestrator(..., debug=debug)
    """

    def __init__(self, file: Any = None, include_timestamps: bool = True):
        self.file = file or sys.stderr
        self.include_timestamps = include_timestamps
        self._request_start: datetime | None = None
        self._shown_message_count: int = 0
        self._last_messages: list[dict[str, str]] = []

    def _timestamp(self) -> str:
        if self.include_timestamps:
            return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
        return ""

    def _print(self, message: str) -> None:
        print(message, file=self.file, flush=True)

    def _print_messages(self, messages: list[dict[str, str]], start: int, truncate: bool = True) -> None:
        for i in range(start, len(messages)):
            msg = messages[i]
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if truncate and len(content) > 500:
                content = content[:500] + "..."
            self._print(f"{self._timestamp()}[{i}] {role}: {content}")

    def on_request(self, messages: list[dict[str, str]], model: str) -> None:
        self._request_start = datetime.now()
        self._last_messages = messages
        self._print(f"{self._timestamp()}━━━ LLM REQUEST ━━━")
        self._print(f"{self._timestamp()}Model: {model}")
        if self._shown_message_count > 0 and self._shown_message_count < len(messages):
            self._print(f"{self._timestamp()}  [... {self._shown_message_count} prior messages]")
        self._print_messages(messages, start=self._shown_message_count, truncate=True)
        self._shown_message_count = len(messages)
        self._print(f"{self._timestamp()}Waiting for response...")

    def on_response(self, response: Any, duration_ms: int) -> None:
        self._print(f"{self._timestamp()}━━━ LLM RESPONSE ({duration_ms:.0f}ms) ━━━")
        try:
            if hasattr(response, "model_dump"):
                self._print(f"{self._timestamp()}{json.dumps(response.model_dump(), indent=2)}")
            else:
                self._print(f"{self._timestamp()}{response}")
        except Exception:
            self._print(f"{self._timestamp()}{response}")

    def on_error(self, error: Exception) -> None:
        lines: list[str] = []
        if self._last_messages:
            lines.append(f"{self._timestamp()}━━━ CONVERSATION TRANSCRIPT ━━━")
            for i, msg in enumerate(self._last_messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"{self._timestamp()}[{i}] {role}: {content}")
        lines.append(f"{self._timestamp()}━━━ ERROR ━━━")
        lines.append(f"{self._timestamp()}{type(error).__name__}: {error}")
        self._print("\n".join(lines))
