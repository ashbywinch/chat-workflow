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
        "'everyone is happy', and not just what it looks like.",
        "The consumer is a person, role, or team — not the artifact itself. "
        "What happens to it next says who receives it and what they do with it, "
        "or that it stays with the user and is not handed on. Where it is kept "
        "says where. None of these restates what it contains.",
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
        description="Who uses or receives it (a person, role, or team); name the party only — what they do with it belongs in integration_points",
        min_length=1,
    )
    format: Annotated[
        str,
        UserLabel("what it looks like", "format"),
    ] = Field(
        ...,
        description="What it looks like: its exact layout (e.g. 'one-page PDF', 'shared spreadsheet')",
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

    @model_validator(mode="after")
    def _integration_is_not_the_consumer(self) -> Deliverable:
        """What happens next is not a repeat of who receives it."""
        if (
            self.integration_points.strip().casefold()
            == self.consumer.strip().casefold()
        ):
            raise ValueError(
                "integration_points must say what happens to it next — who "
                "receives it and what they do with it — not repeat the consumer"
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

    The conversation has already opened. If it does not already name the thing the user makes, ask "What do you make?" in your first turn; if it does, use that name and go straight to proposing what it looks like. Do not greet again and do not re-ask for a name the user has already given. Then reflect back what they said and propose what it looks like and what it holds. Give them a draft to correct, not a form to fill in. If they name something common (meeting minutes, invoices), propose what it usually looks like in that same turn, before asking anything else. Ask only for what is still missing or uncertain, once each.

    Ask for a check they can apply the moment the thing exists, not an outcome: "Every action item has an owner and a due date", not "Everyone is happy". If they give an outcome, propose a concrete check from your domain knowledge and ask them to confirm. Keep the user's own words for a check: if they say "10 minutes of standing", say that, not the standard cooking term for it.

    For who gets it next, say what they do with it — read it, act on their action items, approve it — not only who they are. For where it is kept, name the place.

    A question about how to organise the thing, or what headings it should have, is part of what it looks like: answer with one concrete recommendation. If they ask what you think or say "you tell me", give the suggestion for the point under discussion and end the reply there; do not ask about another point in that reply. If they ask you to write the thing itself, say in one sentence that you are helping them decide what it looks like; if every part is decided, finish and return the result then, and never come back to a point they already confirmed.

    Stay at the level of the thing they make, not its contents. If they describe their domain (dishes, articles, marketing), acknowledge it in a clause and return to what it looks like, using their word for it.

    When they confirm, return the result. Do not summarise it again or ask for a second approval."""
        ...  # type: ignore[reportReturnType]
