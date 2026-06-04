from __future__ import annotations

from pydantic import Field, model_validator

from chat_workflow import InteractiveEntity, ValidationError, atomic_workflow


class Letter(InteractiveEntity):
    _validation_rules: str = "The letter should flow well and achieve the author's aims"

    to: str = Field(..., description="Who is the letter addressed to")
    body: str = Field(..., description="The body of the letter")

    @model_validator(mode="after")
    def validate_business_rules(self):
        if not self.The:
            raise ValidationError("The letter should flow well and achieve the author's aims")
        return self

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        context: str,
        max_turns: int = 10,
        session=None,
    ):
        """Letter workflow."""
        ...
