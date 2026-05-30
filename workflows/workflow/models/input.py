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

        Ask one question at a time.
        """
        ...  # type: ignore[reportReturnType]