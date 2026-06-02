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
    """Open-ended exploration of the user's process.

    What process would you like to work on?
    Tell me about how you currently do [process].
    What does good look like?

    Explore what the user does currently. Listen and absorb.
    Let the user describe their process in their own words.
    Do NOT propose structure or try to define phases and activities yet.
    """
    ...  # type: ignore[reportReturnType]


@atomic_workflow(conversation_validation_rules=[ONE_GUESS, SYNTHESIZES_HONESTLY, NO_EXECUTOR_MODE])
def _generate_from_notes(
    notes: Annotated[str, "The raw notes gathered during exploration"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Propose what the process looks like based on what the user described.

    When the user hasn't mentioned something, suggest ONE possibility at a time.

    Use plain, natural language. You are having a conversation, not writing a report.
    NEVER use model field names like "phases", "activities", "orchestrating_component",
    or "participants" with the user. Instead ask: "what are the main stages?",
    "what happens in each stage?", "who or what makes it happen?"

    Propose what you can based on what the user told you. Summarize what you've heard:
    "So from what you've described, it sounds like there are three stages: deciding
    what you want to eat, checking what you can make, and writing up the plan.
    Is that right?"

    If the user is confused, simplify your language.
    Never put fabricated values in the final output. Propose ideas and let the user
    confirm or correct.
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
