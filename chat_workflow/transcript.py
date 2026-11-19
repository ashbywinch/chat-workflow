"""What the model was told and what it said, read from the stored messages.

The transcript *is* PydanticAI's message list: thinking, tool calls, tool
returns, retry prompts and the final structured output all live there, and
``ModelMessagesTypeAdapter`` round-trips it byte-for-byte. This module is the
one projection of that list into text — for the debug stream, for a failure
message, and for a judge — plus the serialisation used by anything that stores
a run.

Two views, one source. The audit view (``audience="all"``) includes the
model's thinking, its tool traffic and the rejected outputs a retry prompt
answered. The user view keeps only what a person saw: their own turns and the
assistant's replies, with a marker where the validated result was produced.
Nothing is inferred from message text, and no second history exists.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .models import ANSWER_TOOL

Audience = Literal["all", "user"]

# Entry kinds. ``output`` is the final structured response (the request's
# schema made it, the caller validated it); ``structured-result`` is the typed
# value the workflow produced.
_USER_KINDS = frozenset({"user", "assistant"})


def entries(messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
    """The transcript as ``(kind, text)``, in the order the model saw it.

    The last response's text is ``output``: it is the structured answer, not a
    conversational turn, and a caller decides how to show it.
    """
    last_answer = _last_answer_index(messages)
    result: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        if isinstance(message, ModelRequest):
            result.extend(_request_entries(message))
        elif isinstance(message, ModelResponse):
            result.extend(_response_entries(message, is_answer=index == last_answer))
    return result


def render(
    messages: Sequence[ModelMessage],
    *,
    audience: Audience = "all",
    result: Any = None,
) -> str:
    """The transcript as text, or the subset a person saw.

    ``result`` is the validated value the workflow produced. With it, the raw
    structured answer is not repeated as a turn; the result takes its place.
    Without it (a failed run), the raw answer stays — it is the evidence of
    what went wrong.
    """
    parts: list[tuple[str, str]] = []
    for kind, text in entries(messages):
        if kind == "output":
            # In the audit view a failed run keeps its last answer as evidence.
            # A judge never sees it: the raw envelope is internal.
            if audience == "all" and result is None:
                parts.append((kind, text))
            continue
        if audience == "user" and kind not in _USER_KINDS:
            continue
        parts.append((kind, text))
    if result is not None:
        rendered = (
            result.model_dump_json(indent=2)
            if hasattr(result, "model_dump_json")
            else json.dumps(result, indent=2, default=str)
        )
        if audience == "user":
            parts.append(("structured-result", "(the workflow produced its validated result)"))
        else:
            parts.append(("structured-result", rendered))
    return "\n---\n".join(f"[{kind}]\n{text}" for kind, text in parts)


def dump(messages: Sequence[ModelMessage]) -> bytes:
    """The stored form: the message list, byte-for-byte."""
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def load(data: bytes | str) -> list[ModelMessage]:
    """Read a transcript written by :func:`dump`."""
    return ModelMessagesTypeAdapter.validate_json(data)


def _last_answer_index(messages: Sequence[ModelMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelResponse) and any(
            (isinstance(part, TextPart) and part.content.strip())
            or (isinstance(part, ToolCallPart) and part.tool_name == ANSWER_TOOL)
            for part in message.parts
        ):
            return index
    return None


def _request_entries(message: ModelRequest) -> Iterable[tuple[str, str]]:
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            # The instructions are the same every turn; the step's own record
            # of them is the prompt, not a transcript entry.
            continue
        if isinstance(part, UserPromptPart):
            yield "user", _text(part.content)
        elif isinstance(part, RetryFeedbackPart):
            yield "validation", _text(part.content)
        elif isinstance(part, ToolReturnPart):
            if part.tool_name == ANSWER_TOOL:
                # the answer itself, already shown as its call
                continue
            outcome = getattr(part, "outcome", "success")
            kind = "validation" if outcome == "retried" else "tool-return"
            yield kind, _tool_return_text(part)


def _response_entries(message: ModelResponse, *, is_answer: bool) -> Iterable[tuple[str, str]]:
    for part in message.parts:
        if isinstance(part, TextPart):
            if part.content.strip():
                yield ("output" if is_answer else "assistant"), part.content
        elif isinstance(part, ThinkingPart):
            yield "thinking", part.content
        elif isinstance(part, ToolCallPart):
            if part.tool_name == ANSWER_TOOL:
                yield ("output" if is_answer else "assistant"), part.args_as_json_str()
            else:
                yield "tool-call", f"{part.tool_name}({part.args_as_json_str()})"


def _tool_return_text(part: ToolReturnPart) -> str:
    content = part.content
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _text(content: Any) -> str:
    """A user or feedback part's text, even when it carries rich content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(item if isinstance(item, str) else str(item) for item in content)
    return str(content)
