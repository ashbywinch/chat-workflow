from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_analysis import ProcessAnalysis


class Input(BaseModel):
    """A single workflow input."""

    source: str = Field(..., description="Where this input originates")
    format: str = Field(..., description="Exact format/structure")
    trigger_conditions: str = Field(
        ..., description="What initiates workflow execution"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required inputs from other components",
    )
    validation_criteria: str = Field(
        ..., description="How to verify input completeness"
    )

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Input]:
        """You are analyzing the inputs required for this business workflow.

        Based on this process analysis, help the user identify what inputs
        are needed to execute the workflow.

        For each input, identify:
        - source: Where this input originates (user, system, component)
        - format: Exact format/structure
        - trigger_conditions: What initiates execution
        - dependencies: Required inputs from other components
        - validation_criteria: How to verify input completeness

        Guide the conversation efficiently:
        - Based on the process analysis, propose the inputs you think are
          needed and share your understanding for validation.
        - Use your domain expertise to infer likely inputs from the process
          description. Offer them as hypotheses for the user to confirm.
        - Never put fabricated values in the final structured output. Only
          include what the user has confirmed. But you can propose ideas
          in conversation.
        - Ask one question at a time. You can share a rich synthesis or
          proposal in your response, but when asking the user for input,
          limit it to a single question per turn.
        """
        ...  # type: ignore[reportReturnType]