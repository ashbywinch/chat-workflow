from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_analysis import ProcessAnalysis


class Output(BaseModel):
    """A single workflow output."""

    consumer: str = Field(
        ..., description="Which components use this output", min_length=1
    )
    format: str = Field(..., description="Exact format/structure", min_length=1)
    success_criteria: str = Field(
        ..., description="How to measure output quality", min_length=1
    )
    integration_points: str = Field(
        ..., description="How outputs connect downstream"
    )
    storage_requirements: str = Field(
        ..., description="Where/how outputs are preserved"
    )

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Output]:
        """You are a workflow analyst helping the user understand what their process produces.

        The user has described their process. Your job is to identify the outputs
        it generates — what gets produced, who uses it, and what makes it good.

        This isn't a form to fill out. You're an expert who has seen many similar
        processes. Use what the user tells you to propose complete ideas for them
        to confirm.

        - When the user confirms an output exists, propose the details yourself.
          For example: "Since meeting notes are used by attendees to remember
          what happened, the consumer would be meeting participants and the format
          would be a structured document. Does that sound right?"
        - Aim to reach a validated list efficiently — propose what you think
          the outputs and their attributes are, then iterate on feedback.
        - If the user provides extra detail, incorporate it rather than asking
          them to confirm it separately.
        - Never put fabricated values in the final output. Only include what
          the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]