# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow, composite_workflow
from chat_workflow.conversation_rules import (
    LISTENS_FIRST,
    NO_EXECUTOR_MODE,
    ONE_GUESS,
    SYNTHESIZES_HONESTLY,
    WARM_OPEN,
)


class ProcessDefinition(BaseModel):
    """Output of the process definition workflow."""

    phases: list[str] = Field(..., description="Logical process phases in order", min_length=1)
    activities: list[str] = Field(..., description="Specific business activities", min_length=1)
    orchestrating_component: str = Field(..., description="Which domain coordinates this workflow", min_length=1)
    participants: list[str] = Field(..., description="All roles/systems involved", min_length=1)


@atomic_workflow(conversation_validation_rules=[WARM_OPEN, LISTENS_FIRST, NO_EXECUTOR_MODE])
def _gather_notes(
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> str:
    """Explore the user's process. Start with: "What do you do?"

    When the user answers, acknowledge what they said and ask for
    more detail: "Tell me more about that." If they give a brief
    answer, say "Anything else?" to encourage elaboration.

    When you have enough detail (3-4 sentences about their process),
    summarize what you heard and mark success. Never repeat the
    same question. Never propose structure or phases.
    """
    ...  # type: ignore[reportReturnType]


@atomic_workflow(conversation_validation_rules=[ONE_GUESS, SYNTHESIZES_HONESTLY, NO_EXECUTOR_MODE])
def _generate_from_notes(
    notes: Annotated[str, "The raw notes gathered during exploration"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Propose a process structure based on what the user described.

    Start by proposing the complete structure based on their notes.
    Suggest the main stages and ask for confirmation — don't ask the
    user to elaborate on each stage individually.

    Use plain language: "what are the main stages?" not "phases".
    Once confirmed, propose the rest one thing at a time.
    Never repeat a question the user already answered.
    Never put fabricated values in the final output.
    """
    ...  # type: ignore[reportReturnType]


@composite_workflow
def generate_from_chat(
    *,
    session: Annotated[Session, "The chat session"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Orchestrate gathering notes and generating a process definition.

    First explores the user's process openly, then synthesizes a structured
    ProcessDefinition from the gathered notes.
    """
    notes = _gather_notes(session=session, max_turns=max_turns)
    return _generate_from_notes(notes=notes, session=session, max_turns=max_turns)


# Import Session at runtime to break circular dependency
from chat_workflow import Session  # noqa: E402
