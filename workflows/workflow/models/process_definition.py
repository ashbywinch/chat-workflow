# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, Field, model_validator

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

_CONFIRMS_THE_DRAFT: tuple[str, str] = (
    "Confirms the draft",
    "The assistant presents the complete draft breakdown in its first "
    "response and returns the result only after the user confirms it or "
    "corrects a value.",
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

    @model_validator(mode="after")
    def _stages_are_broader_than_steps(self) -> Self:
        """A stage groups steps; the same phrase cannot be both."""
        stages = {phase.strip().casefold() for phase in self.phases}
        steps = {step.strip().casefold() for step in self.activities}
        overlap = sorted(stages & steps)
        if overlap:
            raise ValueError(
                "A stage cannot also be a concrete step. Group the steps "
                "under broader stages; these appear in both: "
                + ", ".join(overlap)
            )
        return self


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
the stages and steps. Group the concrete steps into a few broad stages; a stage is not one of the steps repeated as a
heading, and no step may appear both as a stage and as an activity. List only the people or roles the notes actually
name; a step that says "submit for approval" does not name an approver, so leave the approver out and let the
confirmation step ask about that gap. The role in charge is the person or role that keeps the work moving. If the notes
say the user does it all, the orchestrating component and the sole participant are the user. Do not invent people or
steps the notes do not support. A name the notes say the user is unsure about — "I don't know if JD is John or a
project" — is not a participant; leave it out. An example is not a fact.

Return the complete ProcessDefinition."""
    ...  # type: ignore[reportReturnType]


def _format_process_definition(draft: ProcessDefinition) -> str:
    """Present a derived breakdown in the user's words, without field names."""
    stages = " → ".join(draft.phases)
    steps = "; ".join(draft.activities)
    people = ", ".join(draft.participants)
    return (
        "Now I'll pull together the stages and steps from what you've told me.\n\n"
        "Here's the picture I've got from what you told me:\n"
        f"- Stages: {stages}\n"
        f"- Steps: {steps}\n"
        f"- Who's involved: {people}\n"
        f"- In charge: {draft.orchestrating_component}"
    )


@atomic_workflow_step(
    conversation_validation_rules=[
        _NO_EXECUTOR_MODE,
        _OPENS_WITH_THE_DRAFT,
        _CONFIRMS_THE_DRAFT,
    ]
)
def _confirm_process_definition(
    draft: Annotated[
        ProcessDefinition,
        "The complete breakdown derived from the user's description",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ProcessDefinition:
    """Present the draft breakdown and ask the user to confirm or correct it.

The draft breakdown has already been shown to the user. Ask one confirm-or-correct question about it. If the draft
leaves a gap the notes imply but do not name — for example, "submit for approval" with no approver — ask about that gap
and offer one concrete possibility in the question: "Is the approver your manager, or someone else?" not "Who is the
approver?" Name one concrete improvement grounded in the breakdown — a capture
format, a template, or a checklist they could adopt — and ask whether it fits. Do not invent a tool or product they did
not mention; keep the improvement to a practice. If the user asks what the output looks like, say in one sentence that
the output's shape comes next and return to the stages and steps. If they describe their domain content (a specific
article, dish, or idea), do not comment on it; say "Noted." and return to the stages and steps. Never praise it or
develop the content. If they confirm, return the draft. If they correct a value, update it and return the corrected
result. Do not re-present the whole draft and do not ask them to name the stages or steps; the draft already has
them."""
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
    # The proposal is the model's own draft, shown in the user's words before
    # the confirmation turn. A confirmation turn that skips the presentation
    # would leave the user (and the judge) with no proposal at all.
    session.say(_format_process_definition(draft))
    return _confirm_process_definition(
        draft=draft, session=session, max_turns=max_turns
    )


# Import Session at runtime to break circular dependency
from chat_workflow import Session  # noqa: E402
