from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_analysis import ProcessAnalysis


class Input(BaseModel):
    """A single workflow input."""

    source: str = Field(..., description="Where this input originates", min_length=1)
    format: str = Field(..., description="Exact format/structure", min_length=1)
    trigger_conditions: str = Field(
        ..., description="What initiates workflow execution"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required inputs from other components",
    )
    validation_criteria: str = Field(
        ..., description="How to verify input completeness", min_length=1
    )

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Input]:
        """You are a workflow analyst helping the user understand what inputs their process needs.

        The user has described their process and the outputs it produces. Your job is to
        figure out what inputs feed into it — where they come from and what form they take.

        This isn't a form to fill out. You're an expert who has seen many similar
        processes. When the user describes their needs, propose a complete picture
        back to them rather than asking about each input one field at a time.

        - Propose what you think the full set of inputs is with their details,
          then ask the user to confirm or correct. For example: "From what you've
          said, I'm seeing three inputs: meeting notes from the note-taker (free-form
          text), attendee list from the organizer (list format), and previous action
          items from prior minutes. Does that capture everything?"
        - If the user adds or corrects something, update your understanding and
          propose the revised picture — don't ask a follow-up question about each
          correction separately.
        - Never put fabricated values in the final output. Only include what
          the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]