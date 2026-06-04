from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_definition import ProcessDefinition


@dataclass
class IntegrationGap:
    """An unclear handoff or incomplete interface between components."""

    between: str
    description: str


@dataclass
class OwnershipGap:
    """An activity that lacks clear ownership."""

    activity: str
    reason: str


class GapAnalysis(BaseModel):
    """Analysis of missing elements, gaps, and recommendations."""

    _validation_rules: ClassVar[list[str]] = [
        "Entries in missing_components and missing_playbooks must reference "
        "specific domain concepts from the user's business or process, not "
        "generic labels (e.g. 'InvoiceManager', not 'missing component').",
        "Recommendations must be concrete, actionable next steps, not vague "
        "suggestions (e.g. 'Create an InvoiceManager component with input/output "
        "contracts', not 'address the gap').",
    ]

    missing_components: list[str] = Field(
        default_factory=list,
        description="Domain concept names referenced but not yet designed",
        min_length=0,
    )
    missing_playbooks: list[str] = Field(
        default_factory=list,
        description="Playbook names referenced but not yet created",
        min_length=0,
    )
    integration_gaps: list[IntegrationGap] = Field(
        default_factory=list,
        description="Unclear handoffs or incomplete interfaces between components",
    )
    organizational_gaps: list[OwnershipGap] = Field(
        default_factory=list,
        description="Activities or responsibilities without clear ownership",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Concrete actionable steps to address each identified gap",
        min_length=0,
    )

    @atomic_workflow
    @classmethod
    def analyze_from_chat(
        cls,
        components: Annotated[list, "The identified components"],
        analysis: Annotated[ProcessDefinition, "The process definition"],
        existing_components: Annotated[list[str] | None, "List of existing component names"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> GapAnalysis:
        """You help identify what's missing or needs attention in the user's workflow design.

        IMPORTANT RULES:
        - Speak in plain language, not architecture jargon.
        - Focus on practical gaps the user would care about.
        - If everything looks good, indicate no gaps found rather than fabricating issues.
        """
        ...  # type: ignore[reportReturnType]
