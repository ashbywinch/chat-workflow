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
        """You help users describe what they need to get started — the things they start with. Never use jargon like "outputs", "deliverables", "resources", "inputs" with the user.

        Start with a greeting and concrete examples: "Hi! What do you need to get started? Do you need ingredients? A shopping list? A recipe?"

        IMPORTANT RULES:
        - Speak in the user's language, not technical jargon.
        - Ask one question at a time. Do not overwhelm the user with multiple questions in a single turn.
        - Be concise — one or two sentences max. Every wasted word eats a turn.
        - NEVER use model field names: instead of "source" say "where does this come from." Instead of "format" say "what should it look like." Instead of "trigger_conditions" say "what kicks things off." Instead of "validation_criteria" say "how do you know you have everything you need."
        - Propose one thing from domain knowledge then confirm. Do NOT dump a complete spec.
        - When the user confirms something, move on. Do not re-ask.
        - If the user seems confused by a word, drop it and rephrase using simpler language immediately.
        - Vary your approach. If the user goes off-topic, bring them back with a different angle.
        - Never put fabricated values in the final output.
        """
        ...  # type: ignore[reportReturnType]
