"""Confirmation follows the draft, not the other way round.

A confirmation step returns the draft it was given, or a corrected one. An
unchanged draft is confirmed and closes. A changed draft is a new proposal:
the framework shows it in the user's words and asks once more, so the user
never has to confirm a picture they have not seen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class ConfirmationSession(Protocol):
    """What a confirmation needs from a session: show, and speak."""

    def propose(self, result: Any) -> None: ...

    def say(self, text: str) -> None: ...

# One correction round plus its confirmation. A user who keeps changing the
# draft is in a different task, not a confirmation.
_MAX_CHANGES = 2


def _same(left: Any, right: Any) -> bool:
    def dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (list, tuple)):
            return [dump(item) for item in value]
        return value

    return dump(left) == dump(right)


def confirm_draft(
    *,
    session: ConfirmationSession,
    draft: T,
    step: Callable[..., T],
    max_turns: int = 4,
) -> T:
    """Run a confirmation step, presenting every changed draft it returns."""
    current = draft
    for _ in range(_MAX_CHANGES + 1):
        confirmed = step(draft=current, session=session, max_turns=max_turns)
        if _same(confirmed, current):
            return confirmed
        current = confirmed
        # The proposal turn carries its own lead-in and confirm question.
        session.propose(current)
    return current
