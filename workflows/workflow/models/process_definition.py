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
    "content or execute the process for the user.",
)


class ProcessDefinition(LLMValidated):
    """Output of the process definition workflow."""

    _validation_rules: ClassVar[list[str]] = [
        "Phases are broad stages; activities are concrete steps that can be "
        "assigned to a component.",
        "Every activity is concrete enough that a component could own it.",
        "The orchestrating component is the person or role that keeps the "
        "work moving. The user's own role is valid when they do the work end "
        "to end (for example 'Freelance Writer' or 'Meal Planner'). It is not "
        "the name of the overall process ('Meal Planning', 'Content Creation "
        "Process'), a generic label ('Manager', 'Coordinator'), or a software "
        "product the user does not use. A separate coordinator is optional.",
    ]

    phases: Annotated[
        list[str],
        UserLabel("the main stages", "phases"),
    ] = Field(..., description="The main stages of the work, in order", min_length=1)
    activities: Annotated[
        list[str],
        UserLabel("the things you do", "activities"),
    ] = Field(
        ..., description="The specific things that happen in those stages", min_length=1
    )
    orchestrating_component: Annotated[
        str,
        UserLabel(
            "who keeps the work moving",
            "orchestrating component",
            "orchestrating",
        ),
    ] = Field(
        ...,
        description=(
            "The person or role that keeps the work moving; the user's own "
            "role is valid (for example 'Meal Planner'), the process name "
            "('Meal Planning') is not"
        ),
        min_length=1,
    )
    participants: Annotated[
        list[str],
        UserLabel("everyone involved", "participants"),
    ] = Field(
        ..., description="Everyone involved (people, roles, or systems)", min_length=1
    )


@atomic_workflow_step(conversation_validation_rules=[WARM_OPEN, _NO_EXECUTOR_MODE])
def _gather_notes(
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> str:
    """Greet the user, explain that you are going to understand what they do,
    then ask "What do you do?".

    Listen. If they give enough detail — what they do, the main steps, who is
    involved — send a continue message that summarizes what you heard and asks
    if it is right. Propose the stages you heard as a short list: 'It sounds
    like the work has these stages: …' This is a reflection of what they said,
    not a new structure; do not invent stages they did not mention. Ask for
    confirmation once. When they confirm, return that summary as success
    immediately; do not summarize or ask again. If they say "anything else?"
    or ask what comes next, treat it as confirmation and return success. Do
    not ask who keeps the work moving or who the participants are; the next
    step asks those. Stop after the summary is confirmed.

    Stay at the level of how the work happens, not its content. When the user
    describes their domain (articles, dishes, business ideas), acknowledge
    briefly then redirect: 'That sounds interesting — let's focus on how the
    work happens. What are the main stages?' Do NOT join them in discussing
    their domain content.

    Speak in the user's words. Never say "phases" or "activities" to the user;
    say "the main stages" and "the things that happen". Never repeat a question
    the user has already answered; build on their answer instead.

    If the user asks for practical tips or resists the process conversation,
    acknowledge briefly and continue with one concrete question about how the
    work happens. Never end in failure while they are still describing their
    work.
    """
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
def _generate_from_notes(
    notes: Annotated[str, "What the person told you"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """You help someone turn what they told you into a structured breakdown.

    The notes below are what the user already told you and confirmed; do not
    ask them to repeat it or to confirm the same stages again. Start by
    proposing the complete breakdown based on their notes. Ask only about
    details the notes do not yet cover — don't ask the user to elaborate on
    each step individually. If the user doesn't know something, suggest ONE
    possibility and mark it as a suggestion.

    With the user, never say "phases", "activities", "participants", or
    "orchestrating component"; say "the main stages", "the things that
    happen", "everyone involved", and "who keeps the work moving". If the
    user says the role that keeps the work moving is "you" or "me", accept it;
    do not ask for a more specific job title after they answer. Use the user's
    own words for the role; do not replace "me" with a job title like "Meal
    Planner".

    Phases are broad stages, such as Plan, Shop, and Cook. Activities are the
    concrete steps inside them, such as check the fridge, write a list, and
    buy groceries. Do not use the same concrete step as both a phase and an
    activity. `participants` is required: the user plus anyone else involved.

    When the user confirms the breakdown, return success.
    """
    ...  # type: ignore[reportReturnType]


@composite_workflow_step
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
