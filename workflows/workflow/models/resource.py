# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.mixins import LLMValidated

from .deliverable import Deliverable
from .process_definition import ProcessDefinition

_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the process for the user.",
)


class Resource(LLMValidated):
    """A single workflow resource."""

    _validation_rules: ClassVar[list[str]] = [
        "The source says where the material comes from — a person, system, "
        "document, or place — and the format says its shape. A person or "
        "system source is valid (for example 'The organizer' with format "
        "'meeting agenda'). If the source is itself the material (for example "
        "handwritten notes), the format must describe the same kind of "
        "material, not a contradictory one such as a database.",
        "The validation criteria are a check the user can apply when the "
        "resource is produced — specific enough to tell whether it is complete, "
        "not an after-the-fact outcome like 'no one complains', and not a "
        "restatement of how they gather it.",
        "A resource must not list itself as a dependency.",
    ]

    source: Annotated[
        str,
        UserLabel("where it comes from", "source", guard=False),
    ] = Field(
        ..., description="Where it comes from (a person, system, document, or place)", min_length=1
    )
    format: Annotated[
        str,
        UserLabel("what it looks like", "format", guard=False),
    ] = Field(
        ...,
        description=(
            "Its exact shape when it exists (e.g. 'handwritten list', 'CSV "
            "export', 'email thread'), not the action used to gather it"
        ),
        min_length=1,
    )
    dependencies: Annotated[
        list[str],
        UserLabel("what it needs first", "dependencies"),
    ] = Field(
        default_factory=list,
        description="Other things it needs first",
    )
    validation_criteria: Annotated[
        str,
        UserLabel(
            "how you know it is complete",
            "validation criteria",
            "criteria",
        ),
    ] = Field(
        ...,
        description=(
            "A check the user can apply when the resource is produced to know "
            "it is complete — not a restatement of how they gather it and not "
            "an after-the-fact outcome"
        ),
        min_length=1,
    )

    @atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
    @classmethod
    def generate_from_chat(
        cls,
        deliverables: Annotated[list[Deliverable], "What the person makes"],
        analysis: Annotated[
            ProcessDefinition | None,
            "What the person has already told you about how their work happens",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Resource]:
        """Help someone describe what they start with. Open by saying in one
        sentence what you are doing and why, for example: "I want to understand
        what you start with, so it fits how you actually work."
        Match the user's register: use the words they use, and never introduce
        a term for the output shape, a field, or the framework.

        You already know what they make from `deliverables`. Do not ask what they
        are making or what they start with from scratch. Name the output and
        propose the typical things people start with for that kind of work
        (for meeting minutes: handwritten notes, the agenda, a recording),
        then ask which of those they have.

        Some things people start with are inherently whatever the user wrote
        — raw notes, a rough sketch — or are trusted as sent, like an agenda
        from the organiser. Do not ask how they know those are complete; accept
        them as they are and say so. Never ask "how do you know your notes are
        complete?" — notes are whatever the user wrote. Even when the user trusts a material as
        sent, the field still needs a check: use a check on the material itself,
        such as 'The invite lists the date, time, and attendees' or 'The agenda
        covers the topics the organiser sent.' For raw notes, use a minimal
        check such as 'The notes include the meeting date, the attendees, and at
        least one decision or action item.' Do not write 'I trust...' in any
        form as the criterion. Ask the completeness question only where there
        is something to compare against (a checklist, a recording, a prior
        version). For each material, propose what it looks like and a likely
        completeness check from domain knowledge, then ask the user to confirm
        or correct. Do not ask them to describe the format or the check from
        scratch. Work through one material at a time: confirm the first
        material before you describe the next, and never put completeness
        checks for several materials in one message. Every material you
        return needs all four parts — where it comes from, what it looks
        like, what it needs first, and the check that it is complete — so
        when you revise one part, keep the other three. The check must be something they can apply as soon as they
        have the material, not an outcome like 'no one complains'. Never phrase
        a check as a state of mind — 'I know…', 'I've thought of…', 'I
        remember…' — because that cannot be checked; write what is true of the
        material itself: 'Every item on every shelf is listed', not 'I know
        what is on every shelf'. A criterion
        describes the finished material, not the action that produces it:
        convert 'I recall the meals and list them'
        to 'The list includes every meal I have made that fits my equipment
        and restrictions', and 'I check the fridge' to 'The list matches what
        is in the fridge right now'. 'I have looked at every shelf' is an
        action; 'Every shelf and container is checked and every item is
        marked good or bad' is the criterion. The format is what the material
        looks like when it exists — a list, a document, a spreadsheet, a
        recording, a photo — not the action you take to gather it. 'Visual
        and smell inspection' is an action, not a format; the format is the
        inventory list or condition notes that result. If the user says
        'verbal feedback from the sous', the source is 'The sous' and the
        format is 'Verbal feedback'; the criterion is 'Every piece of feedback
        is written on a sticky note or in the feedback log', not 'I cross-check
        with the sous'. If they give a method or an outcome, propose the
        converted check and format in their words and ask them to confirm.
        If it comes from another part of their work, note that it depends on
        that; otherwise it stands alone. Ask about completeness once per
        resource; if they answer, accept it and move on. If they say they
        trust it or do not check it, accept that and never rephrase the same
        completeness question. If they say they have only one starting
        material, accept that and do not ask about other materials again.

        When you summarize or confirm, describe each resource in the user's
        words. Never use 'source', 'format', 'dependencies', or 'validation
        criteria' — those are your labels, not theirs. Never say 'raw
        materials' or 'starting materials'; say 'the things you start with'
        or 'what you already have'. Say where it comes
        from, what it looks like, what it needs first, and, where it applies,
        how they know it is complete. When the user confirms the set, return
        success in that same response; do not summarize the same set again or
        ask for a second confirmation. If the user has already confirmed the
        set in their last message, return success immediately; do not send
        another summary or ask "Is this right?".

        When they ask for ideas, or ask 'what do you think?' or 'suggest a
        format', propose one concrete option in that same reply; do not ask
        another question first, and do not append a new question after the
        suggestion. Make the suggestion and let them respond.

        Stay at the level of the thing they start with, not its contents. When
        the user describes their domain (dishes, ingredients, business ideas),
        acknowledge briefly then redirect: 'That sounds interesting — let's
        focus on what you start with. What kind of information goes into
        that?' Do NOT join them in discussing their domain content.
        """
        ...  # type: ignore[reportReturnType]
