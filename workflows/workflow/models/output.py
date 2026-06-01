from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_analysis import ProcessAnalysis


class Output(BaseModel):
    """A single workflow output."""

    consumer: str = Field(..., description="Which components use this output", min_length=1)
    format: str = Field(..., description="Exact format/structure", min_length=1)
    success_criteria: str = Field(..., description="How to measure output quality", min_length=1)
    integration_points: str = Field(..., description="How outputs connect downstream")
    storage_requirements: str = Field(..., description="Where/how outputs are preserved")

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Output]:
        """You are a workflow analyst helping the user identify the outputs their process produces.
        The user has described their process. Use what they tell you to propose complete outputs
        with all their attributes filled in. Never ask the user to describe a field you can infer
        yourself.

        When the user confirms something, move on. Do not re-ask or re-confirm what was already
        settled. If the user gives you a detailed answer, acknowledge what you learned and decide
        what's still missing — don't repeat the same question.

        For example: "Since meeting notes are used by attendees to remember what happened,
        the consumer would be meeting participants and the format would be a structured document
        for later review. Success criteria would be accuracy and timeliness, and they'd integrate
        with action items tracking. What would you add or change?" That way you propose the full
        picture and the user corrects, rather than asking about each field one at a time.
        """
        ...  # type: ignore[reportReturnType]
