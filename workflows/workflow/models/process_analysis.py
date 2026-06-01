from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow


class ProcessAnalysis(BaseModel):
    """Output of analyze_process()."""

    phases: list[str] = Field(..., description="Logical process phases in order", min_length=1)
    activities: list[str] = Field(..., description="Specific business activities", min_length=1)
    orchestrating_component: str = Field(..., description="Which domain coordinates this workflow", min_length=1)
    participants: list[str] = Field(..., description="All roles/systems involved", min_length=1)

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        process_description: Annotated[str, "A description of the business process to analyze"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ProcessAnalysis:
        """You are a Business Process Analyst & Workflow Architect.

        Your goal is to analyze a business process description and create a
        structured process analysis covering phases, activities, orchestrating
        component, and participants.

        This isn't a form to fill out. Use what the user tells you to propose
        the analysis for them to confirm.

        - When the user describes their process, synthesize and fill in the
          structure yourself. For example: "From what you've described, I'm
          seeing three phases: note-taking, review, and drafting — with
          activities like identifying action items and assigning owners.
          Does that match your understanding?"
        - Never put fabricated values in the final output. Only include what
          the user has confirmed. But you can propose ideas in conversation.
        - If the user's description is genuinely missing critical information,
          ask for it. But if you can make a reasonable inference, offer it
          first and let the user correct you.
        """
        ...  # type: ignore[reportReturnType]
