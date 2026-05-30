from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_analysis import ProcessAnalysis


class Output(BaseModel):
    """A single workflow output."""

    consumer: str = Field(
        ..., description="Which components use this output"
    )
    format: str = Field(..., description="Exact format/structure")
    success_criteria: str = Field(
        ..., description="How to measure output quality"
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
        """You are analyzing the outputs produced by this business workflow.

        Based on this process analysis, help the user identify what outputs
        this workflow produces.

        For each output, identify:
        - consumer: Which components use this output
        - format: Exact format/structure
        - success_criteria: How to measure output quality
        - integration_points: How outputs connect downstream
        - storage_requirements: Where/how outputs are preserved

        Ask one question at a time.
        """
        ...  # type: ignore[reportReturnType]