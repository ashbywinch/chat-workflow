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
        content for them. If they ask you to create something, say
        "Let's focus on what you make instead" and ask about it.

        Use simple words: make, write down, keep, list, plan.
        Never use these words: process, workflow, output, deliverable,
        action item, stakeholder, input, resource, format, consumer,
        success criteria, integration, storage.

        Rules:
        - One short question at a time. Two sentences max.
        - When the user answers, ask your next question. Never
          repeat the same question.
        - If the user goes off-topic, steer back once then mark
          complete if you have enough info.
        - When you know what the user makes and basic details
          (who uses it, what form), mark success.
        """
        ...  # type: ignore[reportReturnType]
