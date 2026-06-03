# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow
from chat_workflow.conversation_rules import NO_EXECUTOR_MODE, NO_FORCED_FIELD_MAPPING, ONE_GUESS

from .deliverable import Deliverable
from .process_definition import ProcessDefinition


class Resource(BaseModel):
    """A single workflow resource."""

    source: str = Field(..., description="Where this resource originates", min_length=1)
    format: str = Field(..., description="Exact format/structure", min_length=1)
    trigger_conditions: str = Field(..., description="What initiates workflow execution")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required resources from other components",
    )
    validation_criteria: str = Field(..., description="How to verify resource completeness", min_length=1)

    @atomic_workflow(conversation_validation_rules=[ONE_GUESS, NO_FORCED_FIELD_MAPPING, NO_EXECUTOR_MODE])
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessDefinition | None, "The process definition, if already available"] = None,
        outputs: Annotated[list[Deliverable] | None, "The deliverables the process should produce, if already known"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Resource]:
        """Listen to the user and help them describe what they start with.

        Propose what they probably start with based on their work.
        For a meeting process: "You'd start with notes and attendee
        lists — right?" For a writing process: "You begin with research
        and drafts — does that sound right?"

        Once they confirm, ask about specifics. Never repeat yourself.
        If they ask for ideas, give suggestions. One topic at a time.
        """
        ...  # type: ignore[reportReturnType]
