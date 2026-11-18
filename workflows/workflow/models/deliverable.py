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
    "content or execute the work for the user.",
)


class Deliverable(LLMValidated):
    """A single workflow deliverable."""

    _validation_rules: ClassVar[list[str]] = [
        "The quality check is one someone can apply when the thing is "
        "produced — not an after-the-fact outcome like 'no one complains' or "
        "'everyone is happy', and not just its shape.",
        "What happens to it next says who receives it and what they do with "
        "it, or that it stays with the user and is not handed on. Where it is "
        "kept says where. Neither restates what it contains.",
        "If the thing's name alone is ambiguous, the explanation says what it "
        "contains.",
    ]

    name: Annotated[
        str,
        UserLabel("what you call it", "name", guard=False),
    ] = Field(
        ...,
        description="What the person calls what they make, e.g. 'Meeting Minutes', 'Invoice PDF'",
        min_length=1,
    )
    description: Annotated[
        str | None,
        UserLabel("what it is", "description", guard=False),
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
        """Describe the one thing the user makes.

    Open with one sentence on what you are doing, then reflect back what they said and propose the shape you can already infer — what it is, who it is for, what it looks like, and what it contains. Give them a draft to correct, not a form to fill in. If they name something common (meeting minutes, invoices), use its standard shape. Ask only for what is still missing or uncertain, once each: how they know it is good enough; who gets it and what happens to it; where it is kept.

    Ask for a check they can apply the moment the thing exists, not an outcome: "Every action item has an owner and a due date", not "Everyone is happy". If they give an outcome, propose a concrete check from your domain knowledge and ask them to confirm.

    A question about how to organise the thing, or what headings it should have, is part of its shape: answer with one concrete recommendation. If they ask what you think, give a concrete suggestion and end that reply with it, not with a question. If they ask you to write the thing itself, say in one sentence that you are helping them pin down its shape, and carry on.

    Stay at the level of the thing they make, not its contents. If they describe their domain (dishes, articles, marketing), acknowledge it in a clause and return to the shape; never call their work "content creation".

    When they confirm, return the result. Do not summarise it again or ask for a second approval."""
        ...  # type: ignore[reportReturnType]
