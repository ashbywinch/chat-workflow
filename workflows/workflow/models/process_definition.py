# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, Field, model_validator

from chat_workflow import atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.confirmation import confirm_draft
from chat_workflow.conversation_rules import WARM_OPEN
from chat_workflow.mixins import LLMValidated
from chat_workflow.rendering import RenderContext

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


class ProcessStage(BaseModel):
    """One broad stage of the work, with the concrete steps inside it."""

    name: Annotated[str, UserLabel("the stage", "stage")] = Field(
        ...,
        description=(
            "A broad stage of the work, named as the user names it (for "
            "example 'Plan', 'Shop', 'Cook')."
        ),
        min_length=1,
    )
    steps: Annotated[list[str], UserLabel("the steps", "steps")] = Field(
        ...,
        description=(
            "The concrete steps inside this stage, in the user's words (for "
            "example 'check the fridge', 'write a list'). A step is not the "
            "stage's name repeated."
        ),
        min_length=1,
    )


class ProcessDefinition(LLMValidated):
    """The stages of the user's work, each with its concrete steps."""

    _validation_rules: ClassVar[list[str]] = [
        "The stages are broad; each stage lists the concrete steps inside it, "
        "and can be assigned to a part of the work.",
        "Every step is concrete enough that one part of the work could "
        "produce it.",
        "No stage name is repeated as one of the steps.",
        "The participants may name the user as 'You'. In this conversation "
        "that is the user's own role, not a placeholder, and it is valid "
        "alongside other named participants.",
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

    stages: Annotated[
        list[ProcessStage],
        UserLabel("the main stages", "stages", guard=False),
    ] = Field(
        ...,
        description=(
            "The broad stages of the work, in order, each with the concrete "
            "steps inside it. A stage is not itself one of those steps."
        ),
        min_length=1,
    )

    @property
    def phases(self) -> list[str]:
        """The stage names, for consumers that need the flat order."""
        return [stage.name for stage in self.stages]

    @property
    def activities(self) -> list[str]:
        """Every step, in stage order, for consumers that need the flat list."""
        return [step for stage in self.stages for step in stage.steps]

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
    def _stages_are_distinct_from_steps(self) -> Self:
        """A stage groups steps; a stage name is not one of the steps."""
        names = [stage.name.strip().casefold() for stage in self.stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Stage names must be distinct; these repeat: "
                + ", ".join(duplicates)
            )
        steps = {step.strip().casefold() for step in self.activities}
        overlap = sorted(set(names) & steps)
        if overlap:
            raise ValueError(
                "A stage cannot also be a concrete step. Put the steps inside "
                "the stage; these appear as both: " + ", ".join(overlap)
            )
        return self


    def render(self, context: RenderContext) -> str:
        """Say the breakdown as a picture, not a field dump."""
        self_words = {"you", "yourself", "me", "i", "myself", "the user", "user"}
        others = [
            person
            for person in self.participants
            if person.strip().casefold() not in self_words
        ]
        if not others:
            people = "You are the only person involved."
        elif len(others) == 1:
            people = f"You and {others[0]} are involved."
        else:
            people = f"You, {_series(others)} are involved."

        owner = self.orchestrating_component.strip()
        keeps = (
            "You keep it moving."
            if owner.casefold() in self_words
            else f"{owner} keeps it moving."
        )

        stages = [
            f"Stage {index} — {stage.name}: {'; '.join(stage.steps)}."
            for index, stage in enumerate(self.stages, 1)
        ]
        return f"The work has {len(stages)} stages. " + " ".join(
            [*stages, f"{people} {keeps}"]
        )


def _series(items: list[str]) -> str:
    """A, B, and C — a sentence, not a list of labels."""
    values = [item.strip() for item in items if item.strip()]
    if len(values) <= 1:
        return values[0] if values else "nothing yet"
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


class GatheredNotes(BaseModel):
    """What the user told us about their work, in their own words."""

    notes: str = Field(
        min_length=1,
        description=(
            "The user's description of how their work happens, in their own "
            "words. Not your reply to them, and never a question."
        ),
    )


@atomic_workflow_step(
    explore=True,
    conversation_validation_rules=[WARM_OPEN, _NO_EXECUTOR_MODE],
)
def _gather_notes(
    process_description: Annotated[
        str, "What the person already told you about how their work happens"
    ] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> GatheredNotes:
    """Collect the user's description of how their work happens.

The conversation has already opened; do not greet again. Say what you are doing. If the description parameter already holds their answer,
it is what they already told you: do not ask "What do you do?" again; ask only "Is there anything else?" once, then return. Otherwise
ask "What do you do?" once; after their first answer, return what they told you, in their words. If their answer is a specific article, dish, or idea rather than what they do, ask one process question — "What
do you actually do, step by step?" — and return their answer.

Listen, and let them talk. Do not name stages, ask them to confirm anything, or structure what they say. This step collects the description; the next step turns it into the breakdown.

Put the user's description in `notes`, in their words — not your reply to them. If you still need an answer, ask the question as a continue; never put a question in `notes`."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_process_definition(
    notes: Annotated[str, "What the person told you, in their words"],
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> ProcessDefinition:
    """Turn the user's description into the stages and steps of their work.

The notes are complete. Derive the breakdown from them yourself; do not ask the user anything. Return each stage with
the steps inside it: `stages: [{name, steps}]`. Keep their names for stages and steps. If the notes name the stages, use
those names and put the actions inside them as steps. When the notes give only steps, group them into a few broad
stages. A stage is not one of its steps repeated as a heading. List only the people or roles the notes actually
name; a step that says "submit for approval" does not name an approver, so leave the approver out and let the
confirmation step ask about that gap. The role in charge is the person or role that keeps the work moving. If the notes
say the user does it all, the orchestrating component and the sole participant are "You"; do not invent a job title
such as "Meal Planner" for the user. Do not invent people or
steps the notes do not support. A name the notes say the user is unsure about — "I don't know if JD is John or a
project" — is not a participant; leave it out. An example is not a fact.

Return the complete ProcessDefinition."""
    ...  # type: ignore[reportReturnType]


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
    """
The draft has been shown. Ask one confirm-or-correct question.

If the draft leaves a gap the notes imply but do not name — for example, "submit for approval" with no approver — ask
about that gap and offer one concrete possibility: "Is the approver your manager, or someone else?" If the user says
they do not handle a step, propose a concrete role for it and ask them to confirm. Name one concrete improvement
grounded in the breakdown — a capture format, a template, or a checklist — and ask whether it fits; keep it to a
practice and do not invent a tool the user did not mention. If the user asks what the output looks like, say in one
sentence that the shape comes next and return to the stages and steps. If they describe domain content, do not comment
on it; say "Noted." and return. If they confirm, return the draft. If they correct a value, update it and return the
corrected result; the framework shows the change and asks for confirmation. Do not re-present the whole draft and do not
ask them to name the stages or steps.
    
    """
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
    # Exploration follows the user's threads: one question per new part of
    # the work, and it stops when they have nothing new. It is not an
    # interrogation, so it is bounded well below the overall turn budget.
    gather_turns = min(max_turns, 6)
    if process_description:
        gathered = _gather_notes(
            process_description=process_description,
            session=session,
            max_turns=gather_turns,
        )
    else:
        gathered = _gather_notes(session=session, max_turns=gather_turns)
    answer = session.io.prompt("Is there anything else?")
    if answer.strip():
        gathered = gathered.model_copy(
            update={"notes": f"{gathered.notes}\n{answer}"}
        )
    draft = _draft_process_definition(notes=gathered.notes, session=session)
    # The proposal is the model's own draft, shown in the user's words before
    # the confirmation turn. A confirmation turn that skips the presentation
    # would leave the user (and the judge) with no proposal at all.
    session.propose(draft)
    return confirm_draft(
        session=session,
        draft=draft,
        step=_confirm_process_definition,
        max_turns=min(max_turns, 4),
    )


# Import Session at runtime to break circular dependency
from chat_workflow import Session  # noqa: E402
