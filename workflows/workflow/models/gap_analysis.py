from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from ..component_responsibilities import ComponentRequirement
from .process_definition import ProcessDefinition


class GapAnalysis(BaseModel):
    """Analysis of missing elements."""

    missing_components: list[str] = Field(..., description="Components referenced but not existing")
    missing_playbooks: list[str] = Field(..., description="Playbooks referenced but not created")
    integration_gaps: list[str] = Field(..., description="Unclear handoffs or incomplete interfaces")
    organizational_gaps: list[str] = Field(..., description="Activities without clear ownership")
    recommendations: list[str] = Field(..., description="How to fill each gap")

    @atomic_workflow
    @classmethod
    def analyze_from_chat(
        cls,
        components: Annotated[list[ComponentRequirement], "The identified components"],
        analysis: Annotated[ProcessDefinition, "The process definition"],
        existing_components: Annotated[list[str] | None, "List of existing component names"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> GapAnalysis:
        """You help identify what's missing or needs attention in the user's workflow design.

        Given the identified building blocks (components), the process analysis, and any
        existing components, check for: missing pieces that haven't been identified,
        connections between pieces that don't quite fit, or organizational concerns.

        IMPORTANT RULES:
        - Speak in plain language, not architecture jargon.
        - Focus on practical gaps the user would care about.
        - If everything looks good, indicate no gaps found rather than fabricating issues.
        """
        ...  # type: ignore[reportReturnType]
