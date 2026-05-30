from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow


class ProcessAnalysis(BaseModel):
    """Output of analyze_process()."""

    phases: list[str] = Field(..., description="Logical process phases in order")
    activities: list[str] = Field(..., description="Specific business activities")
    orchestrating_component: str = Field(
        ..., description="Which domain coordinates this workflow"
    )
    participants: list[str] = Field(
        ..., description="All roles/systems involved"
    )

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        process_description: Annotated[str, "A description of the business process to analyze"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ProcessAnalysis:
        """You are a Business Process Analyst & Workflow Architect.

        Your goal is to analyze a business process description and create a
        structured process analysis.

        Behavior:
        - Collaborate with the user to understand their business objective
        - Identify logical process phases in order
        - Identify specific business activities within each phase
        - Determine the appropriate orchestrating component (which domain coordinates this)
        - Identify all participants (roles, systems, components)
        - Validate your understanding with the user
        - Ask one question at a time
        - Base your analysis only on information explicitly provided by the user
        - If the user is vague, ask clarifying questions
        """
        ...  # type: ignore[reportReturnType]