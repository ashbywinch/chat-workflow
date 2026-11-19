# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.conversation_rules import WARM_OPEN
from chat_workflow.mixins import LLMValidated

_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the work for the user.",
)

_OPENS_WITH_THE_DRAFT: tuple[str, str] = (
    "Opens with the draft",
    "The assistant's first response presents the complete breakdown derived "
    "from the notes, in the user's words. It does not ask the user to name the "
    "stages, the steps, the role, or the participants that the notes already "
    "establish.",
)


class ProcessDefinition(LLMValidated):
    """The stages and concrete steps of the user's work."""

    _validation_rules: ClassVar[list[str]] = [
        "The stages are broad; the steps are concrete and can be assigned "
        "to a part of the work.",
        "Every step is concrete enough that one part of the work could "
        "produce it.",
        "The person in charge is the person or role that keeps the work moving. The user's own role is valid when they do the work end "
        "to end (for example 'Freelance Writer' or 'Meal Planner'); a job "
        "title the user states, such as 'Project Manager', is their own role "
        "and is accepted. It is not "
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
            "who is in charge",
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


class GatheredNotes(BaseModel):
    """What the user told us about their work, in their own words."""

    notes: str = Field(
        min_length=1,
        description=(
            "The user's description of how their work happens, in their own "
            "words. Not your reply to them, and never a question."
        ),
    )


@atomic_workflow_step(conversation_validation_rules=[WARM_OPEN, _NO_EXECUTOR_MODE])
def _gather_notes(
    process_description: Annotated[
        str, "What the person already told you about how their work happens"
    ] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> GatheredNotes:
    """Collect the user's description of how their work happens.

The conversation has already opened; do not greet again. Say what you are doing. If the description parameter already holds their answer,
it is what they already told you: do not ask "What do you do?" again; ask only "Is there anything else?" once, then return. Otherwise
ask "What do you do?" once; after their first answer, ask at most "Is there anything else?" once, then return what they told you, in
their words.

Listen, and let them talk. Do not name stages, ask them to confirm anything, or structure what they say. This step collects the description; the next step turns it into the breakdown.

Put the user's description in `notes`, in their words — not your reply to them. If you still need an answer, ask the question as a continue; never put a question in `notes`."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_process_definition(
    notes: Annotated[str, "What the person told you, in their words"],
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> ProcessDefinition:
    """Turn the user's description into the stages and steps of their work.

The notes are complete. Derive the breakdown from them yourself; do not ask the user anything. Keep their names for
the stages and steps. The role in charge is the person or role that keeps the work moving; if the notes say the user
does it all, the orchestrating component and the sole participant are the user. Do not invent people or steps the notes
do not support.

Return the complete ProcessDefinition."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE, _OPENS_WITH_THE_DRAFT])
def _confirm_process_definition(
    draft: Annotated[
        ProcessDefinition,
        "The complete breakdown derived from the user's description",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Present the draft breakdown and ask the user to confirm or correct it.

The draft is the complete breakdown derived from their description. Present it in their words — the stages, the
steps inside them, who is involved, and who is in charge — then ask one confirm-or-correct question. Do not ask them to
name the stages or steps; the draft already has them. If they correct a value, update it and return the corrected
result. If they confirm, return it. Ask only for a gap the draft cannot fill; do not re-ask what the draft contains."""
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
    # Gathering is a short step by contract: ask once, invite elaboration
    # once, return. A larger budget here only lets the model ask again.
    gather_turns = min(max_turns, 4)
    if process_description:
        gathered = _gather_notes(
            process_description=process_description,
            session=session,
            max_turns=gather_turns,
        )
    else:
        gathered = _gather_notes(session=session, max_turns=gather_turns)
    draft = _draft_process_definition(notes=gathered.notes, session=session)
    return _confirm_process_definition(
        draft=draft, session=session, max_turns=max_turns
    )


# Import Session at runtime to break circular dependency
from chat_workflow import Session  # noqa: E402
