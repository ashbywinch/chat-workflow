from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .input import Input
from .output import Output
from .process_analysis import ProcessAnalysis


class ComponentRequirement(BaseModel):
    """A component identified as needed by the workflow."""

    name: str = Field(..., description="Artifact-based name (noun)", min_length=1)
    purpose: str = Field(..., description="Single-sentence purpose", min_length=1)
    required_inputs: list[str] = Field(
        ..., description="Input names from input analysis"
    )
    expected_outputs: list[str] = Field(
        ..., description="Output names from output analysis"
    )
    component_type: str = Field(
        ...,
        description="One of: value_stream, artifact_producing, planning_service",
        min_length=1,
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
        """You are a business architect helping the user identify the components their process needs.

        The user has described their process, inputs, and outputs. Your job is to
        identify the distinct business components that make up the workflow.

        When the user tells you about their process, propose complete components
        with name, purpose, and type. Use your expertise to fill in the details.

        For example: "Based on the meeting minutes process, I'm seeing a Notes
        artifact component (artifact_producing) for recording meeting discussions,
        and a Minutes Draft component (artifact_producing) for turning notes into
        formal minutes."

        Never put fabricated values in the final output. Only include what
        the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]