from __future__ import annotations

from pydantic import Field

from chat_workflow import InteractiveEntity, atomic_workflow_step

# lucidlint: ignore-file class-module Letter is this package's one model; models.py is the package's model home


class Letter(InteractiveEntity):
    _validation_rules: str = "The letter should flow well and achieve the author's aims"

    to: str = Field(..., description="Who is the letter addressed to")
    body: str = Field(..., description="The body of the letter")

    @atomic_workflow_step
    @classmethod
    def generate_from_chat(
        cls,
        context: str,
        session=None,
    ):
        """Letter workflow."""
        ...
