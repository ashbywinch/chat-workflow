from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from ..component_responsibilities import ComponentRequirement
from .process_analysis import ProcessAnalysis


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
        analysis: Annotated[ProcessAnalysis, "The process analysis"],
        existing_components: Annotated[list[str] | None, "List of existing component names"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> GapAnalysis:
        """You are analyzing gaps in the component architecture.

        Review the required components against what already exists.
        Identify:
        - missing_components: Components referenced but not existing
        - missing_playbooks: Playbooks referenced but not created
        - integration_gaps: Unclear handoffs or incomplete interfaces
        - organizational_gaps: Activities without clear ownership
        - recommendations: How to fill each gap

        Be thorough. Every activity in the process analysis must have clear
        component ownership. Every gap must have a concrete recommendation.
        """
        ...  # type: ignore[reportReturnType]
