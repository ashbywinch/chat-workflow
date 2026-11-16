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
        "Process'), a label that is only generic ('Manager', 'Coordinator'), "
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
            "role is valid (for example 'Meal Planner'), the process name "
            "('Meal Planning') is not. A bare 'Coordinator' is too generic; "
            "'Meeting Minutes Coordinator' names what it coordinates and is "
            "specific enough."
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
        str, "What the person already told you about their process"
    ] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> str:
    """Find out how the user's work happens.

    Greet them, say what you are doing, and ask "What do you do?". If
    `process_description` already holds their answer, start from it instead.

    When they answer, name the stages their work breaks into: the ones they
    described, plus the stage this kind of work usually has when theirs is
    thin, marked as a suggestion. Ask them to correct the list, and return it
    once they confirm it.

    Ask one question at a time, about the work rather than its content. If they
    describe their domain, acknowledge it in a clause and return to the stages.
    Do not ask who is involved or who keeps the work moving: the next step does.
    """
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
def _generate_from_notes(
    notes: Annotated[str, "What the person told you"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Build the breakdown from the stages the user confirmed.

    The notes hold those stages. Do not present them again.

    Work out the concrete steps inside each stage from what they told you, and
    their order, keeping the names the user used. Ask about a stage only when
    they gave no detail for it at all. Ask who is involved and who keeps things
    moving if the notes do not say; if they are clear, return at once. If the
    user does not know something, suggest one possibility and mark it as a
    suggestion. When the user gives "me" or "the person" as the role, that is
    the answer.

    When they confirm the breakdown, return it.
    """
    ...  # type: ignore[reportReturnType]


@composite_workflow_step
def generate_from_chat(
    *,
    session: Annotated[Session, "The chat session"],
    process_description: Annotated[
        str, "What the person already told you about their process"
    ] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Orchestrate gathering notes and generating a process definition.

    First explores the user's process openly, then synthesizes a structured
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
