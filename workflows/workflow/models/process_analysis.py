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
        structured process analysis covering:
        - Logical process phases in order
        - Specific business activities within each phase
        - The orchestrating component (which domain coordinates this)
        - All participants (roles, systems, components)

        Guide the conversation efficiently:
        - When the user describes their process, synthesize what they've said
          into the analysis fields and share your understanding for validation.
          For example: "From what you've described, I'm seeing three phases:
          note-taking, review, and drafting — with activities like identifying
          action items and assigning owners. Does that match your understanding?"
        - Use your domain expertise to interpret what the user tells you and fill
          in the structure. Reasonable inferences are fine — offer them as
          hypotheses for the user to confirm or correct.
        - Never put fabricated values in the final structured output. Only include
          what the user has confirmed. But you can propose ideas in conversation.
        - If the user's description is genuinely missing critical information,
          ask for it. But if you can make a reasonable inference, offer it first
          and let the user correct you.
        - Aim to reach a validated analysis efficiently — share your understanding
          early and iterate on feedback.
        - Ask one question at a time. You can share a rich synthesis or proposal
          in your response, but when asking the user for input, limit it to a
          single question per turn.
        """
        ...  # type: ignore[reportReturnType]