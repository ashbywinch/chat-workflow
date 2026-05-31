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

        For each component, identify:
        - name: Artifact-based name (noun), not a process (verb)
        - purpose: Single-sentence purpose
        - required_inputs: Input names from input analysis
        - expected_outputs: Output names from output analysis
        - component_type: One of value_stream, artifact_producing, planning_service

        Guide the conversation efficiently:
        - Based on the process analysis and the inputs/outputs, propose the
          components you think are needed and share your understanding for
          validation. For example: "Based on the meeting minutes process, I'm
          seeing three components needed: a Notes artifact, a Minutes draft,
          and an Action Items tracker. Does that align with what you need?"
        - Use your domain expertise to infer likely components. Offer them
          as hypotheses for the user to confirm or correct.
        - Never put fabricated values in the final structured output. Only
          include what the user has confirmed. But you can propose ideas
          in conversation.
        - Ask one question at a time. You can share a rich synthesis or
          proposal in your response, but when asking the user for input,
          limit it to a single question per turn.
        """
        ...  # type: ignore[reportReturnType]