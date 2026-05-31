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

        Guide the conversation efficiently:
        - Based on the process analysis, propose the outputs you think are
          produced and share your understanding for validation.
        - Use your domain expertise to infer likely outputs from the process
          description. Offer them as hypotheses for the user to confirm.
        - Never put fabricated values in the final structured output. Only
          include what the user has confirmed. But you can propose ideas
          in conversation.
        - Ask one question at a time. You can share a rich synthesis or
          proposal in your response, but when asking the user for input,
          limit it to a single question per turn.
        """
        ...  # type: ignore[reportReturnType]