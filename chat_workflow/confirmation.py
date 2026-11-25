"""Confirmation follows the draft, not the other way round.

A confirmation step returns the draft it was given, or a corrected one. An
unchanged draft is confirmed and closes. A changed draft is a new proposal:
the framework shows it in the user's words and asks once more, so the user
never has to confirm a picture they have not seen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from .rendering import RenderContext, render_result

__all__ = ["ConfirmationSession", "confirm_draft"]

T = TypeVar("T")


class ConfirmationSession(Protocol):
    """What a confirmation needs from a session: show, and speak."""

    def propose(
        self,
        result: Any,
        *,
        changed: bool = False,
        note: str | None = None,
    ) -> None: ...

    def say(self, text: str) -> None: ...


def _rendered(value: Any) -> str:
    return render_result(value, RenderContext())


def _same(left: Any, right: Any) -> bool:
    # The user sees the rendering, not the model dump. A change that does not
    # alter a word of the proposal is not a new proposal.
    return _rendered(left) == _rendered(right)


def confirm_draft(
    *,
    session: ConfirmationSession,
    draft: T,
    step: Callable[..., T],
    max_turns: int = 4,
    # One correction round plus its confirmation. A user who keeps changing the
    # draft is in a different task, not a confirmation.
    max_changes: int = 2,
) -> T:
    """Run a confirmation step, presenting every changed draft it returns."""
    current = draft
    for _ in range(max_changes + 1):
        confirmed = step(draft=current, session=session, max_turns=max_turns)
        if _same(confirmed, current):
            return confirmed
        current = confirmed
        # The proposal turn carries its own lead-in and confirm question.
        session.propose(current, changed=True)
    return current
