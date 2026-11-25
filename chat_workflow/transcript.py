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
from typing import Any, Literal, NamedTuple

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

from .models import AgentAnswer
from .session_log import TranscriptSegment

__all__ = ["AnswerEnvelope", "TranscriptEntry", "dump", "entries", "load", "render"]

Audience = Literal["all", "user"]


class TranscriptEntry(NamedTuple):
    """One line of the transcript: what kind of turn, and its text."""

    kind: str
    text: str


class AnswerEnvelope(NamedTuple):
    """The answer body a tool call carries."""

    intent: str | None = None
    message: str | None = None

# Entry kinds. ``output`` is the final structured response (the request's
# schema made it, the caller validated it); ``structured-result`` is the typed
# value the workflow produced.
_USER_KINDS = frozenset({"user", "assistant"})

# The framework speaks first and last for a step that has no user of its own.
# These prompts reach the model as user turns, but a person never said them, so
# the user view does not show them and a judge is not asked to read them.
_FRAMEWORK_PROMPTS = frozenset(
    {
        "Start the conversation.",
        "No user is available. Produce the result now from the instructions "
        "and the inputs.",
        "The conversation is over. Close it in one sentence.",
        "The user has finished. Say in one sentence that you have what they "
        "need.",
        "The user has finished. Say in one sentence that you have what they "
        "need. ",
        "Describe the result to the user in their words.",
        "You are finishing too early. Do the task's next step before returning "
        "the result.",
        "No user is available to answer. Return success with the completed "
        "result, or failure if it cannot be produced.",
        "A success message cannot ask the user a question. If you still need "
        "an answer, return intent='continue'. Otherwise return the result "
        "with a statement of what you did.",
    }
)


# The prompts the framework injects after a turn it refuses: an early success,
# a success that asks, or a chat-off continue. The refused turn was
# never delivered to the user, so the user view drops it.
_CORRECTION_PROMPTS = frozenset(
    {
        "You are finishing too early. Do the task's next step before returning "
        "the result.",
        "A success message cannot ask the user a question. If you still need "
        "an answer, return intent='continue'. Otherwise return the result "
        "with a statement of what you did.",
        "No user is available to answer. Return success with the completed "
        "result, or failure if it cannot be produced.",
        "Your message ends with a colon but the turn does not contain the "
        "content it introduces. Give the content in the message, or finish "
        "and let the next step show it.",
        "You returned failure. Continue with what you were given: propose the "
        "result, or ask the one question you still need. Failure is only for a "
        "user who refuses to engage or input that cannot be interpreted.",
    }
)


_INTERNAL_PROMPTS = _FRAMEWORK_PROMPTS | _CORRECTION_PROMPTS


def _has_correction_prompt(message: ModelRequest) -> bool:
    return any(
        isinstance(part, UserPromptPart)
        and _text(part.content).strip() in _CORRECTION_PROMPTS
        for part in message.parts
    )


def entries(
    messages: Sequence[ModelMessage],
    *,
    suppress_corrected: bool = False,
) -> list[TranscriptEntry]:
    """The transcript as ``(kind, text)``, in the order the model saw it.

    The last response's text is ``output``: it is the structured answer, not a
    conversational turn, and a caller decides how to show it. With
    ``suppress_corrected``, a response the framework refused (it was followed
    by a correction prompt) is dropped: the user never saw that turn.
    """
    suppressed: set[int] = set()
    if suppress_corrected:
        last_response: int | None = None
        for index, message in enumerate(messages):
            if isinstance(message, ModelResponse):
                last_response = index
            elif (
                isinstance(message, ModelRequest)
                and last_response is not None
                and _has_correction_prompt(message)
            ):
                suppressed.add(last_response)
    last_answer = _last_answer_index(messages)
    result: list[TranscriptEntry] = []
    for index, message in enumerate(messages):
        if index in suppressed:
            continue
        if isinstance(message, ModelRequest):
            result.extend(_request_entries(message))
        elif isinstance(message, ModelResponse):
            result.extend(_response_entries(message, is_answer=index == last_answer))
    # An output retry can leave the same assistant text twice in a row: the
    # rejected attempt and the accepted one. A person saw it once, so the
    # transcript shows it once.
    deduped: list[TranscriptEntry] = []
    _dedupe_assistant_repeats(deduped, result)
    return deduped


def _dedupe_assistant_repeats(deduped, result):
    for kind, text in result:
        if (
            deduped
            and kind == "assistant"
            and deduped[-1][0] == "assistant"
            and deduped[-1][1].strip() == text.strip()
        ):
            continue
        deduped.append(TranscriptEntry(kind=kind, text=text))


def render(
    messages: Sequence[ModelMessage],
    *,
    audience: Audience = "all",
    result: Any = None,
    segments: Sequence[TranscriptSegment] | None = None,
) -> str:
    """The transcript as text, or the subset a person saw.

    ``result`` is the validated value the workflow produced. With it, the raw
    structured answer is not repeated as a turn; the result takes its place.
    Without it (a failed run), the raw answer stays — it is the evidence of
    what went wrong.

    ``segments`` marks each step's slice of the concatenated transcript. The
    user view needs it: a session log holds several steps, each with its own
    final answer, and only the step's own answer may be dropped as internal.
    A chat-off step's turns are dropped entirely — no user saw them.
    """
    if audience == "user" and segments:
        parts = _user_segment_entries(messages, segments)
    else:
        parts = _conversation_entries(messages, audience, result)
    if result is not None:
        parts.append(_structured_result_entry(result, audience))
    return "\n---\n".join(f"[{kind}]\n{text}" for kind, text in parts)


def _user_segment_entries(
    messages: Sequence[ModelMessage],
    segments: Sequence[TranscriptSegment],
) -> list[TranscriptEntry]:
    # Steps cover slices of the log; the composite's own turns (echoes and
    # spoken messages) sit in the gaps between them and must stay: they are
    # exactly what the user saw happen between steps.
    ranges: list[tuple[int, int, bool]] = []
    cursor = 0
    for segment in segments:
        if segment.start > cursor:
            ranges.append((cursor, segment.start, True))
        ranges.append((segment.start, segment.end, segment.chat))
        cursor = segment.end
    if cursor < len(messages):
        ranges.append((cursor, len(messages), True))
    parts: list[TranscriptEntry] = []
    for start, end, chat in ranges:
        if not chat:
            continue
        for kind, text in entries(messages[start:end], suppress_corrected=True):
            if kind == "output":
                continue
            if kind == "user" and text.strip() in _INTERNAL_PROMPTS:
                continue
            parts.append(TranscriptEntry(kind=kind, text=text))
    return parts


def _conversation_entries(
    messages: Sequence[ModelMessage],
    audience: Audience,
    result: Any,
) -> list[TranscriptEntry]:
    parts: list[TranscriptEntry] = []
    for kind, text in entries(messages, suppress_corrected=audience == "user"):
        if kind == "output":
            # In the audit view a failed run keeps its last answer as
            # evidence. A judge never sees it: the raw envelope is internal.
            if audience == "all" and result is None:
                parts.append(TranscriptEntry(kind=kind, text=text))
            continue
        if audience == "user" and kind not in _USER_KINDS:
            continue
        if audience == "user" and kind == "user" and text.strip() in _INTERNAL_PROMPTS:
            continue
        parts.append(TranscriptEntry(kind=kind, text=text))
    return parts


def _structured_result_entry(result: Any, audience: Audience) -> TranscriptEntry:
    if audience == "user":
        return TranscriptEntry(
            kind="structured-result",
            text="(the workflow produced its validated result)",
        )
    rendered = (
        result.model_dump_json(indent=2)
        if hasattr(result, "model_dump_json")
        else json.dumps(result, indent=2, default=str)
    )
    return TranscriptEntry(kind="structured-result", text=rendered)


def dump(messages: Sequence[ModelMessage]) -> bytes:
    """The stored form: the message list, byte-for-byte."""
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def load(data: bytes | str) -> list[ModelMessage]:
    """Read a transcript written by :func:`dump`."""
    return ModelMessagesTypeAdapter.validate_json(data)


def _answer_envelope(part: ToolCallPart) -> AnswerEnvelope | None:
    args = part.args
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    body = args.get("response", args)
    if not isinstance(body, dict):
        return None
    return AnswerEnvelope(intent=body.get("intent"), message=body.get("message"))


def _is_final_answer(envelope: AnswerEnvelope | None) -> bool:
    return envelope is not None and envelope.intent in ("success", "failure")


def _text_is_an_envelope(text: str) -> bool:
    """True for the raw JSON of a success/failure answer, however it arrived.

    A provider sometimes returns the envelope as an escaped string inside
    ``response``, so both the plain and the escaped key forms count.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    has_intent = '"intent"' in stripped or '\\"intent\\"' in stripped
    has_outcome = any(
        marker in stripped
        for marker in (
            '"success"',
            '"failure"',
            '\\"success\\"',
            '\\"failure\\"',
        )
    )
    return has_intent and has_outcome


def _last_answer_index(messages: Sequence[ModelMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if (
                isinstance(part, ToolCallPart)
                and part.tool_name == AgentAnswer.TOOL_NAME
                and _is_final_answer(_answer_envelope(part))
            ):
                return index
            if isinstance(part, TextPart) and _text_is_an_envelope(part.content):
                return index
    return None


def _request_entries(message: ModelRequest) -> Iterable[TranscriptEntry]:
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            # The instructions are the same every turn; the step's own record
            # of them is the prompt, not a transcript entry.
            continue
        if isinstance(part, UserPromptPart):
            yield TranscriptEntry(kind="user", text=_text(part.content))
        elif isinstance(part, RetryFeedbackPart):
            yield TranscriptEntry(kind="validation", text=_text(part.content))
        elif isinstance(part, ToolReturnPart):
            if part.tool_name == AgentAnswer.TOOL_NAME:
                # the answer itself, already shown as its call
                continue
            outcome = getattr(part, "outcome", "success")
            kind = "validation" if outcome == "retried" else "tool-return"
            yield TranscriptEntry(kind=kind, text=_tool_return_text(part))


def _response_entries(message: ModelResponse, *, is_answer: bool) -> Iterable[TranscriptEntry]:
    for part in message.parts:
        if isinstance(part, TextPart):
            if part.content.strip():
                # A rejected attempt's envelope is as internal as the accepted
                # one: no user ever saw either.
                kind = (
                    "output"
                    if _text_is_an_envelope(part.content) or is_answer
                    else "assistant"
                )
                yield TranscriptEntry(kind=kind, text=part.content)
        elif isinstance(part, ThinkingPart):
            yield TranscriptEntry(kind="thinking", text=part.content)
        elif isinstance(part, ToolCallPart):
            if part.tool_name == AgentAnswer.TOOL_NAME:
                envelope = _answer_envelope(part)
                spoken = envelope.message if envelope else None
                # A continue answer is the assistant's question and is shown;
                # a success/failure answer is the structured result and is not.
                kind = "output" if _is_final_answer(envelope) else "assistant"
                yield TranscriptEntry(kind=kind, text=spoken or part.args_as_json_str())
            else:
                yield TranscriptEntry(kind="tool-call", text=f"{part.tool_name}({part.args_as_json_str()})")


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
