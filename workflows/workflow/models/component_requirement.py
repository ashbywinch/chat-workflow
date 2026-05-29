from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .input import Input
from .output import Output
from .process_analysis import ProcessAnalysis


class ComponentRequirement(BaseModel):
    """A component identified as needed by the workflow."""

    name: str = Field(..., description="Artifact-based name (noun)")
    purpose: str = Field(..., description="Single-sentence purpose")
    required_inputs: list[str] = Field(
        ..., description="Input names from input analysis"
    )
    expected_outputs: list[str] = Field(
        ..., description="Output names from output analysis"
    )
    component_type: str = Field(
        ...,
        description="One of: value_stream, artifact_producing, planning_service",
    )

    @atomic_workflow
    @classmethod
    def identify_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis to identify components from"],
        inputs: Annotated[list[Input], "The workflow inputs"],
        outputs: Annotated[list[Output], "The workflow outputs"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[ComponentRequirement]:
        """You are a Business Architect identifying components needed for this process.

        {analysis}

        Rules:
        - Each component is named after a business artifact (noun), not a process (verb)
        - Each component has a single, clear responsibility
        - Prefer specific components over vague "coordination" or "monitoring"
        - Distinguish between existing and proposed components
        - Every input must be consumed by at least one component
        - Every output must be produced by at least one component
        - Ask one question at a time
        - Base your analysis only on information explicitly provided by the user
        """
        ...  # type: ignore[reportReturnType]