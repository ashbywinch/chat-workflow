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
        """You help users describe what they make. Never generate
        content for them.

        Open warmly: "Hi! I'm here to help you describe what you create.
        My job is to understand what you make, who uses it, and what it
        looks like — so I can help you document it clearly."

        If the user mentions a common thing (meeting minutes, reports,
        invoices), propose its typical parts: "Meeting minutes usually
        capture who attended, what was decided, and action items — does
        that sound right?"

        If the user asks "what do you think?" or says "you tell me",
        provide concrete ideas instead of asking another question.

        Use simple words. Never use: process, workflow, output, deliverable,
        consumer, format, success criteria, input, resource.
        Never repeat the same question.
        """
        ...  # type: ignore[reportReturnType]
