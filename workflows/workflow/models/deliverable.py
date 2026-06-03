# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow
from chat_workflow.conversation_rules import NO_EXECUTOR_MODE, ONE_GUESS, SYNTHESIZES_HONESTLY

from .process_definition import ProcessDefinition


class Deliverable(BaseModel):
    """A single workflow deliverable."""

    consumer: str = Field(..., description="Which components use this deliverable", min_length=1)
    format: str = Field(..., description="Exact format/structure", min_length=1)
    success_criteria: str = Field(..., description="How to measure deliverable quality", min_length=1)
    integration_points: str = Field(..., description="How deliverables connect downstream")
    storage_requirements: str = Field(..., description="Where/how deliverables are preserved")

    @atomic_workflow(conversation_validation_rules=[ONE_GUESS, SYNTHESIZES_HONESTLY, NO_EXECUTOR_MODE])
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessDefinition | None, "The process definition, if already available"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Deliverable]:
        """Listen to the user. Help them describe what they make.

        Open: "Hi! I'm here to help you describe what you create so
        I can understand and document it."

        Then ask: "Who uses what you make?" and "What does it look
        like?" If they mention meeting minutes, propose: "Minutes
        usually have attendees, decisions, action items — right?"

        When the user asks for ideas, give them. Don't repeat yourself.
        """
        ...  # type: ignore[reportReturnType]
