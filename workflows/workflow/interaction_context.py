"""ComponentInteractionContext model — domain considerations for assistant interaction."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .domain_spec import ComponentDomainSpec
from .structure import ComponentStructure


class ComponentInteractionContext(BaseModel):
    """Domain considerations for how the assistant interacts with the user
    during artifact creation.

    This captures what the assistant should prioritize, suggest, and watch
    for — pure domain language, not code or prompt implementation.
    """

    must_prioritize: list[str] = Field(
        ...,
        description=(
            "What the assistant should always address during the conversation, "
            "e.g. 'Always ask about decisions early in the conversation'"
        ),
    )
    auto_suggest: list[str] = Field(
        ...,
        description=(
            "What the assistant should suggest proactively to the user, "
            "e.g. 'Suggest action item owners based on the topic discussed'"
        ),
    )
    tone_preference: str | None = Field(
        default=None,
        description=(
            "Tone guidance for the assistant's communication style, "
            "e.g. 'Professional but friendly'. None means no preference."
        ),
    )
    user_pain_points: list[str] = Field(
        ...,
        description=("Common pitfalls the assistant should watch for, e.g. 'Users often forget to list attendees'"),
    )

    @atomic_workflow
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
        """You are a conversation designer. The user's artifact is already
        defined; now help them define how the assistant should behave.

        Ask the user what they expect from the assistant. For example:
        "What should I always capture or do when helping you? What should
        I suggest before you ask? What mistakes should I watch for?"

        After they answer, propose your understanding back as a complete
        picture and ask for confirmation. For example: "From what you've
        told me, it sounds like you want me to always capture decisions
        with context, suggest owners proactively, keep a professional
        tone, and watch for vague descriptions. Does that sound right?"

        Use "continue" to ask or propose, "success" only when confirmed.
        Never include both message and result. Stay in domain language.
        """
        ...  # type: ignore[reportReturnType]
