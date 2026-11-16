"""The message list a chat model receives, with its constraints enforced.

A chat transcript is stricter than it looks. The rules below are documented by
the providers, or measured on the model this project is tuned for. They live
here rather than in a comment at the call site so that getting them wrong is a
failure where the mistake is made, not a confusing empty completion later.

What each role means
--------------------
``system``
    An instruction *to* the model, from the application. It is not a turn and
    it is not something the model said. Putting a turn here makes it part of
    the model's standing brief; putting an instruction in ``assistant`` makes
    the model believe it produced that text.
``user``
    A turn from the person. A framework correction is a user turn too — it is
    the next thing the model is asked to respond to — and carries the
    ``[framework]`` prefix so it is never mistaken for the user's own words.
``assistant``
    The model's own output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]

# The model this project is tuned for returns an empty completion when the
# request ends on a system message: 4 empty replies in 6 with the instruction
# last, 0 in 6 with it first, same conversation.
_EMPTY_ON_SYSTEM_LAST = (
    "Measured on this model: the same conversation with the instruction last "
    "returned an empty completion 4 times in 6, and 0 times in 6 with it first."
)


class TranscriptError(RuntimeError):
    """The message list is not in a shape a chat API accepts."""


@dataclass
class Transcript:
    """A conversation assembled so the provider can answer it.

    Two rules are applied silently because they are provider-compatibility
    normalisations, not mistakes: multiple instructions are folded into one
    leading system message, and consecutive ``user`` turns are joined. Every
    other violation raises, with the rule and its source next to the raise.
    """

    instructions: list[str] = field(default_factory=list)
    turns: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: Iterable[dict[str, Any]]) -> Transcript:
        transcript = cls()
        for message in messages:
            transcript.append(
                str(message.get("role") or ""), str(message.get("content") or "")
            )
        return transcript

    def append(self, role: str, content: str) -> None:
        """Add one message, or raise if it cannot be part of a valid request."""
        if not content.strip():
            raise TranscriptError(
                "A message may not be empty: several providers reject an empty "
                "text block, and one consisting only of whitespace too."
            )
        if role == "system":
            if self.turns:
                raise TranscriptError(
                    "A system message must come before the conversation starts. "
                    "It is an instruction to the model, not a turn, so a system "
                    "message after a turn is in the wrong place. "
                    + _EMPTY_ON_SYSTEM_LAST
                )
            self.instructions.append(content)
            return
        if role not in ("user", "assistant"):
            raise TranscriptError(
                f"Unknown role {role!r}. The transcript uses system, user and "
                "assistant; tool results would need their own pairing rules."
            )
        if self.turns and self.turns[-1]["role"] == role:
            if role == "assistant":
                raise TranscriptError(
                    "Two assistant turns in a row: the model did not reply "
                    "between them. Some providers merge consecutive same-role "
                    "turns silently and others reject the request, so the "
                    "caller must not produce this."
                )
            # Consecutive user turns are common (a turn, then a framework
            # correction). Providers merge them, so join them here.
            self.turns[-1]["content"] += "\n\n" + content
            return
        self.turns.append({"role": role, "content": content})

    def user(self, content: str) -> None:
        self.append("user", content)

    def assistant(self, content: str) -> None:
        self.append("assistant", content)

    def wire(self) -> list[dict[str, str]]:
        """The list to send, or raise if the model would have nothing to answer."""
        if not self.turns:
            raise TranscriptError(
                "The request has no conversation turn. A chat request needs a "
                "user turn to respond to; a system-only request is answered "
                "with an empty completion. " + _EMPTY_ON_SYSTEM_LAST
            )
        if self.turns[-1]["role"] != "user":
            raise TranscriptError(
                "The request must end with a user turn: 'To trigger a "
                "response from the model, end with a user message' (OpenAI "
                "chat completions). Ending on an assistant turn is a prefill, "
                "which current models reject outright; ending on a system "
                "message leaves nothing to answer."
            )
        messages: list[dict[str, str]] = []
        if self.instructions:
            messages.append({"role": "system", "content": "\n\n".join(self.instructions)})
        messages.extend(self.turns)
        return messages
