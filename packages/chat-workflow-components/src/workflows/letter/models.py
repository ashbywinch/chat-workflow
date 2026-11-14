from __future__ import annotations

from pydantic import Field

from chat_workflow import InteractiveEntity, atomic_workflow_step


class Letter(InteractiveEntity):
    _validation_rules: str = "The letter should flow well and achieve the author's aims"

    to: str = Field(..., description="Who is the letter addressed to")
    body: str = Field(..., description="The body of the letter")

    @atomic_workflow_step
    @classmethod
    def generate_from_chat(
        cls,
        context: str,
        max_turns: int = 10,
        session=None,
    ):
        """Letter workflow."""
        ...
