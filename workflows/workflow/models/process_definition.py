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
    """Explore what the user does. Ask once, listen, then summarize.

    "What do you do?" — then listen. If they give enough detail,
    say "Got it, let me summarize what I heard..." and use success.
    Never ask about the same thing twice. Stay open-ended.
    """
    ...  # type: ignore[reportReturnType]


@atomic_workflow(conversation_validation_rules=[ONE_GUESS, SYNTHESIZES_HONESTLY, NO_EXECUTOR_MODE])
def _generate_from_notes(
    notes: Annotated[str, "The raw notes gathered during exploration"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Propose a process structure based on what the user described.

    Start by proposing the complete structure based on their notes.
    Suggest the main steps and ask for confirmation — don't ask the
    user to elaborate on each step individually.

    Use plain language: "steps" or "parts", not "stages" or "phases".
    Once confirmed, propose the rest one thing at a time.
    If the user doesn't know something, suggest ONE possibility.
    Never ask the same question twice. Never fabricate values.
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
