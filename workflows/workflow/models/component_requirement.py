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

        This isn't a form to fill out. You're an expert who has seen many similar
        processes. Interpret what the user tells you and fill in the structure.
        Reasonable inferences are fine — offer them as hypotheses for the user to
        confirm or correct.

        - When the user describes their process, propose the components you see.
          For example: "Based on the meeting minutes process, I'm seeing three
          components: a Notes artifact, a Minutes draft, and an Action Items
          tracker. Does that align with what you need?"
        - Aim to reach a validated list efficiently — propose what you think
          the components are and iterate on feedback rather than asking the user
          to describe every component from scratch.
        - Never put fabricated values in the final output. Only include what
          the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]