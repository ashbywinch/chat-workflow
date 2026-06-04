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
        """You are a conversation designer. Sequence is:
        1. Ask what the assistant should prioritize.
        2. After answer, ask about proactive suggestions.
        3. After answer, ask about common mistakes.
        4. After answer, propose the complete picture back.
        5. User confirms → success.

        Return "continue" for steps 1-4. Return "success" at step 5
        only. Never return success before step 5.
        """
        ...  # type: ignore[reportReturnType]
