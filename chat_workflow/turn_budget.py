"""The one turn budget a session shares across its steps.

``max_turns`` is a runaway guard for the whole session, not a per-step
allowance. Every model turn takes from this counter; when it is exhausted the
step raises :class:`~chat_workflow.exceptions.TurnLimitExceededError`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import TurnLimitExceededError

__all__ = ["TurnBudget"]


@dataclass
class TurnBudget:
    """A session-wide model-turn budget.

    ``None`` means no bound. The counter is shared by every session in the
    tree, so a composite cannot reset or multiply it by calling more steps.
    """

    max_turns: int | None
    turn_count: int = 0

    @property
    def exhausted(self) -> bool:
        return self.max_turns is not None and self.turn_count >= self.max_turns

    def take(self) -> bool:
        """Consume one turn; return True when it was the last allowed turn."""
        if self.max_turns is not None and self.turn_count >= self.max_turns:
            raise TurnLimitExceededError(self.max_turns)
        self.turn_count += 1
        return self.max_turns is not None and self.turn_count >= self.max_turns
