# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field, model_validator

from chat_workflow import atomic_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.mixins import LLMValidated

from .process_definition import ProcessDefinition

_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the process for the user.",
)


class Deliverable(LLMValidated):
    """A single workflow deliverable."""

    _validation_rules: ClassVar[list[str]] = [
        "The success criteria are a check someone can apply when the "
        "deliverable is produced — not an after-the-fact outcome like 'no one "
        "complains' or 'everyone is happy', and not just its format.",
        "The integration points say what happens to it after it is made: "
        "who receives it next and what they do with it, or that it stays "
        "with the user and is not handed on. The storage requirements say "
        "where it is kept. Neither restates what it contains.",
        "If the name alone is ambiguous, the description explains what the "
        "deliverable contains.",
    ]

    name: Annotated[
        str,
        UserLabel("what you call it", "name"),
    ] = Field(
        ...,
        description="What the person calls what they make, e.g. 'Meeting Minutes', 'Invoice PDF'",
        min_length=1,
    )
    description: Annotated[
        str | None,
        UserLabel("what it is", "description"),
    ] = Field(
        default=None,
        description="Plain-English description of what it contains — "
        "required unless the name alone is unambiguous (e.g. 'Invoice' is clear; "
        "'Widget' is not).",
    )
    consumer: Annotated[
        str,
        UserLabel("who it is for", "consumer"),
    ] = Field(
        ...,
        description="Who uses or receives it (a person, role, or team)",
        min_length=1,
    )
    format: Annotated[
        str,
        UserLabel("what it looks like", "format"),
    ] = Field(
        ...,
        description="Its exact shape or layout (e.g. 'one-page PDF', 'shared spreadsheet')",
        min_length=1,
    )
    success_criteria: Annotated[
        str,
        UserLabel("how you know it is good", "success criteria", "criteria"),
    ] = Field(
        ...,
        description=(
            "A check someone can apply when the deliverable is produced — not "
            "an after-the-fact outcome and not just its format"
        ),
        min_length=1,
    )
    integration_points: Annotated[
        str,
        UserLabel("who gets it next", "integration points"),
    ] = Field(
        ...,
        description="Who receives it next, and what they do with it",
        min_length=1,
    )
    storage_requirements: Annotated[
        str,
        UserLabel("where it is kept", "storage requirements"),
    ] = Field(
        ...,
        description="Where it is kept",
        min_length=1,
    )

    @model_validator(mode="after")
    def _consumer_is_a_party(self) -> Deliverable:
        """The consumer is a person, role, or team — not the artifact itself."""
        if self.consumer.strip().casefold() == self.name.strip().casefold():
            raise ValueError(
                "consumer must be a person, role, or team, not the deliverable itself"
            )
        return self

    @atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[
            ProcessDefinition | None,
            "What the person has already told you about how their work happens",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Deliverable]:
        """Help someone describe what they make. Say in one sentence what you
        are doing, then reflect back what they said. If they name something
        common (meeting minutes, invoices), propose its standard shape
        immediately as a draft for confirmation — do not ask who it is for
        before offering the shape.

        Never use failure because the user says they make nothing. Find the
        artifact in their routine — a plan, a list, a schedule, a document —
        and propose it as a draft: 'It sounds like you make a weekly meal
        plan; does that fit?' Use failure only if they reject every concrete
        proposal.

        Match the user's register: use the words they use, and never introduce
        a term for the output shape, a field, or the framework. Never say
        'deliverable' to the user; say 'the thing you make'. If they speak
        plainly, speak plainly; if they use specialist terms, use theirs. Ask
        who it is for, what it looks like, how they know it is good enough,
        who it is handed to next and what happens then, and where it is kept —
        in whatever words fit the user. Make sure the criterion can be checked
        the moment the thing exists, rather than waiting for an outcome like
        'no one complains' or 'everyone is happy'. If the user gives an
        outcome, propose a concrete check from your domain knowledge (for
        minutes: every decision has a reason, every action item has an owner
        and due date, the attendee list is complete) and ask them to confirm.

        Ask each question once. When the user answers, accept the answer and
        move on; never ask the same substantive question again.

        When you summarize or confirm, describe the thing they make in the
        user's words. Never use 'name', 'description', 'consumer', 'format',
        'success criteria', 'integration points', or 'storage requirements' —
        those are your labels, not theirs. Say what it is, who it is for, what
        it looks like, how they know it is good, who gets it next, and where
        it is kept. Write a short paragraph; never a labelled list. Do not
        make them fill in a form.

        Stay at the level of the thing they make, not its content. When the
        user describes their domain (business ideas, dishes, articles),
        acknowledge briefly then redirect: 'That sounds interesting — let's
        focus on what you make. What does the finished thing look like?' Do
        NOT join them in discussing their domain content.

        When they ask for 'what do you think?' or 'suggest something',
        PROPOSE one concrete idea from your domain knowledge — do NOT ask more
        questions or say 'how would you like to structure it'.
        """
        ...  # type: ignore[reportReturnType]
