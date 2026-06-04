# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow
from chat_workflow.conversation_rules import NO_EXECUTOR_MODE, ONE_GUESS, SYNTHESIZES_HONESTLY

from .process_definition import ProcessDefinition


class Deliverable(BaseModel):
    """A single workflow deliverable."""

    name: str = Field(
        ...,
        description="Name of the deliverable, e.g. 'Meeting Minutes', 'Invoice PDF'",
        min_length=1,
    )
    description: str | None = Field(
        default=None,
        description="Plain-English description of what this deliverable contains — "
        "required unless the name alone is unambiguous (e.g. 'Invoice' is clear; "
        "'Widget' is not).",
    )
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
        """Help someone describe what they create. Greet warmly and
        explain your purpose before anything else. If they mention
        something common (meeting minutes, invoices), suggest its
        standard structure from your expertise.

        Keep the conversation in their language, not business terms.
        When they ask for ideas, suggest them. Don't repeat questions.
        Return success when you have enough for the model.
        """
        ...  # type: ignore[reportReturnType]
