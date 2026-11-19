"""ComponentInteractionContext model — domain considerations for assistant interaction."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step
from chat_workflow.annotations import UserLabel

from .domain_spec import ComponentDomainSpec
from .structure import ComponentStructure


class ComponentInteractionContext(BaseModel):
    """Domain considerations for how the assistant interacts with the user
    during artifact creation.

    This captures what the assistant should prioritize, suggest, and watch
    for — pure domain language, not code or prompt implementation.
    """

    must_prioritize: Annotated[
        list[str],
        UserLabel("what I should always cover", "must prioritize"),
    ] = Field(
        ...,
        description=(
            "What the assistant should always address during the conversation, "
            "e.g. 'Always ask about decisions early in the conversation'"
        ),
    )
    auto_suggest: Annotated[
        list[str],
        UserLabel("what I should suggest", "auto suggest"),
    ] = Field(
        ...,
        description=(
            "What the assistant should suggest proactively to the user, "
            "e.g. 'Suggest action item owners based on the topic discussed'"
        ),
    )
    tone_preference: Annotated[
        str | None,
        UserLabel("how I should sound", "tone preference"),
    ] = Field(
        default=None,
        description=(
            "Tone guidance for the assistant's communication style, "
            "e.g. 'Professional but friendly'. None means no preference."
        ),
    )
    user_pain_points: Annotated[
        list[str],
        UserLabel("what usually goes wrong", "user pain points"),
    ] = Field(
        ...,
        description=("Common pitfalls the assistant should watch for, e.g. 'Users often forget to list attendees'"),
    )

    @atomic_workflow_step
    @classmethod
    def gather(
        cls,
        domain_spec: Annotated[
            ComponentDomainSpec,
            (
                "The domain specification — what the artifact means in the user's"
                " world, its fields, and what makes it excellent"
            ),
        ],
        structure: Annotated[
            ComponentStructure,
            ("The structural design — what fields and rules the artifact follows"),
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ComponentInteractionContext:
        """You are a conversation designer. Start from the domain spec and
        structure; do not ask the user to define the interaction from scratch.
        They already name the artifact, so never ask what to call it.
        Your first message must propose all four parts:
        1. What the assistant should always cover.
        2. What it should suggest proactively.
        3. The tone, only if the user volunteered one.
        4. What usually goes wrong that the assistant should catch.

        Even if the user has already volunteered some or all of the answers,
        restate the complete picture as one proposal and ask them to confirm
        it; do not return success on the first turn. Derive likely answers for
        all four from the artifact's domain and structure, present them, and
        then ask the user to confirm or correct each one with explicit
        questions: "What should I always prioritize?
        What would you like me to suggest proactively? What usually goes wrong
        that I should catch?" Do not ask only about priorities, and do not ask
        each topic as a separate turn. If the user asks to go deeper on one
        point, answer that point and return to the proposal. When the user
        confirms the proposal, return success. If the user refuses to engage,
        return failure.
        """
        ...  # type: ignore[reportReturnType]
