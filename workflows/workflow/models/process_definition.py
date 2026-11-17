# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.conversation_rules import WARM_OPEN
from chat_workflow.mixins import LLMValidated

_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the work for the user.",
)


class ProcessDefinition(LLMValidated):
    """The stages and concrete steps of the user's work."""

    _validation_rules: ClassVar[list[str]] = [
        "Phases are broad stages; activities are concrete steps that can be "
        "assigned to a component.",
        "Every activity is concrete enough that a component could own it.",
        "The orchestrating component is the person or role that keeps the "
        "work moving. The user's own role is valid when they do the work end "
        "to end (for example 'Freelance Writer' or 'Meal Planner'). It is not "
        "the name of the work as a whole ('Meal Planning', 'Content Creation'), "
        "a label that is only generic ('Manager', 'Coordinator'), "
        "or a software product the user does not use. A role that names what "
        "it coordinates is specific ('Meeting Minutes Coordinator'), not "
        "generic. When the user names their own role as 'me', 'I', or "
        "'myself', that is their answer and is accepted; do not demand a job "
        "title. A separate coordinator is optional.",
    ]

    phases: Annotated[
        list[str],
        UserLabel("the main stages", "phases", guard=False),
    ] = Field(
        ...,
        description=(
            "The broad stages of the work, in order, named as the user names "
            "them (for example 'Plan', 'Shop', 'Cook'). A stage groups several "
            "concrete steps; it is not itself one of those steps."
        ),
        min_length=1,
    )
    activities: Annotated[
        list[str],
        UserLabel("the things you do", "activities", guard=False),
    ] = Field(
        ...,
        description=(
            "The concrete steps inside those stages, in the user's words (for "
            "example 'check the fridge', 'write a list', 'buy groceries'). No "
            "step may appear both here and as a stage."
        ),
        min_length=1,
    )
    orchestrating_component: Annotated[
        str,
        UserLabel(
            "who keeps things moving",
            "orchestrating component",
            "orchestrating",
            guard=False,
        ),
    ] = Field(
        ...,
        description=(
            "The person or role that keeps the work moving; the user's own "
            "role is valid (for example 'Meal Planner'), the name of the work "
            "as a whole ('Meal Planning') is not. A bare 'Coordinator' is too "
            "generic; 'Meeting Minutes Coordinator' names what it coordinates "
            "and is specific enough."
        ),
        min_length=1,
    )
    participants: Annotated[
        list[str],
        UserLabel("everyone involved", "participants", guard=False),
    ] = Field(
        ..., description="Everyone involved (people, roles, or systems)", min_length=1
    )


@atomic_workflow_step(conversation_validation_rules=[WARM_OPEN, _NO_EXECUTOR_MODE])
def _gather_notes(
    process_description: Annotated[
        str, "What the person already told you about how their work happens"
    ] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> str:
    """Collect the user's description of how their work happens.

Greet them, say what you are doing, and ask "What do you do?". If the description parameter already holds their answer,
start from it instead.

Listen, and let them talk. Ask at most one follow-up — "Is there anything else?" — then return what they told you, in
their words.

Do not name stages, ask them to confirm anything, or structure what they say. This step collects the description; the next step turns it into the breakdown."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
def _generate_from_notes(
    notes: Annotated[str, "What the person told you"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Turn the user's description into the stages and steps of their work, and confirm it.

The notes hold what the user told you. Derive the breakdown from them yourself: the stages, the concrete steps inside
each, everyone involved, and who keeps things moving. Do not walk the user through the stages asking what happens in
each — work it out from their description and present the result as a draft, keeping their own names.

Present the complete breakdown and ask them to confirm it. Ask only for what the description does not contain at all:
who is involved, and who keeps things moving. If the user does not know something, suggest one possibility and mark it
as a suggestion. When they give "me" or "the person" as the role, that is the answer.

When they confirm, return it."""
    ...  # type: ignore[reportReturnType]


@composite_workflow_step
def generate_from_chat(
    *,
    session: Annotated[Session, "The chat session"],
    process_description: Annotated[
        str, "What the person already told you about how their work happens"
    ] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Orchestrate gathering notes and generating the stages and steps of the work.

    First explores how the user works openly, then synthesizes a structured
    ProcessDefinition from the gathered notes. When `process_description` is
    provided, start from it instead of asking the user to repeat it.
    """
    if process_description:
        notes = _gather_notes(
            process_description=process_description,
            session=session,
            max_turns=max_turns,
        )
    else:
        notes = _gather_notes(session=session, max_turns=max_turns)
    return _generate_from_notes(notes=notes, session=session, max_turns=max_turns)


# Import Session at runtime to break circular dependency
from chat_workflow import Session  # noqa: E402
