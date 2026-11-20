"""Rendering a typed result for the person reading the conversation.

A step's result is internal: it is what the caller returns. The proposal and
summary turns are for a person, so the record itself renders them — it knows
its own shape and the words the user uses. The framework owns the contract:

- **surface** — a line-oriented chat transcript, or a terminal. Both are plain
  text; the chat has no tables, colour, or markup it cannot display.
- **audience** — the user, or a developer reading a debug view. The user view
  never contains a schema field name or a Python type expression.
- **guardrail** — ``render_result`` validates every rendering before it can
  reach the transcript: no control characters, no HTML, no markdown the chat
  cannot show, bounded line length, and (for the user) none of the internal
  vocabulary below.

Every render is for the user. The renderer prints the record's own values and
fixed user-facing labels; it must not add schema labels of its own, but it must
not censor the user's words either — a user's domain may legitimately contain
"consumer" or "format". The guard therefore checks **display**, not
vocabulary. Each result model implements :class:`Renderable`; a list of
records renders item by item. A step that allows an empty list renders nothing,
and its own success message says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class RenderSurface(StrEnum):
    """Where the text is displayed."""

    CHAT = "chat"
    CLI = "cli"


class RenderError(ValueError):
    """A rendering that would display incorrectly."""


@dataclass(frozen=True)
class RenderContext:
    """Where the text is displayed, and how wide a line may be.

    The audience is always the user: a developer reads the transcript, not a
    rendering. A record that needs another record's words carries them itself
    (``StructField.field_type_hint``) rather than reaching through a context.
    """

    surface: RenderSurface = RenderSurface.CHAT
    max_line: int = 120


@runtime_checkable
class Renderable(Protocol):
    """A record that can describe itself to a person."""

    def render(self, context: RenderContext) -> str: ...


def plain_type(type_expr: Any) -> str:
    """Say a type expression in business words, not Python syntax."""
    text = str(type_expr).strip()
    optional = text.endswith("| None") or text.startswith("Optional[")
    text = text.replace("| None", "").strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1]
    if text == "str":
        plain = "text"
    elif text in {"int", "float"}:
        plain = "a number"
    elif text == "bool":
        plain = "yes or no"
    elif text.startswith("list[") and text.endswith("]"):
        plain = "a list of " + plain_type(text[len("list[") : -1])
    elif text.startswith("dict[") or text == "dict":
        plain = "a set of values"
    else:
        spaced = "".join(
            f" {character}" if character.isupper() else character
            for character in text
        )
        plain = spaced.replace("_", " ").strip().casefold()
    return plain + (" (optional)" if optional else "")


def render_result(value: Any, context: RenderContext) -> str:
    """Render a step's typed result, or ``""`` when there is nothing to show."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        blocks: list[str] = []
        for item in value:
            if not isinstance(item, Renderable):
                raise RenderError(
                    f"{type(item).__name__} in a list does not implement "
                    "Renderable; every item in a user-facing proposal renders "
                    "itself."
                )
            blocks.append(validate_rendering(item.render(context), context))
        return "\n".join(block for block in blocks if block)
    if isinstance(value, Renderable):
        return validate_rendering(value.render(context), context)
    raise RenderError(
        f"{type(value).__name__} does not implement Renderable; a user-facing "
        "proposal needs a render() on the record."
    )


def _wrap_line(line: str, width: int) -> str:
    """Wrap one line at spaces, keeping its indentation on continuation lines."""
    indent = line[: len(line) - len(line.lstrip())]
    continuation = indent + ("  " if line.lstrip().startswith("- ") else "")
    words = line.split()
    if not words:
        return line
    lines: list[str] = []
    current = indent
    for word in words:
        candidate = f"{current} {word}" if current.strip() else indent + word
        if len(candidate) <= width:
            current = candidate
            continue
        if not current.strip():
            raise RenderError(
                f"A single word is {len(word)} characters; the limit is {width}."
            )
        lines.append(current.rstrip())
        current = continuation + word
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def validate_rendering(text: str, context: RenderContext) -> str:
    """Wrap long lines and reject text that would display incorrectly."""
    if not text:
        return ""
    text = "\n".join(_wrap_line(line, context.max_line) for line in text.splitlines())
    if "\x1b" in text or "\r" in text:
        raise RenderError("A rendering must not carry control characters.")
    if re.search(r"<[a-zA-Z/]", text):
        raise RenderError("A rendering must not contain HTML.")
    if "**" in text or "`" in text:
        raise RenderError("A rendering must not contain markdown emphasis or code.")
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            raise RenderError("A rendering must not contain markdown headings.")
        if "|" in line:
            raise RenderError(
                "A rendering must not contain a table; the chat cannot align one."
            )
    return text
