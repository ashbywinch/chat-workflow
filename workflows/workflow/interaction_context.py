"""ComponentInteractionContext model — domain considerations for assistant interaction."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step

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
        """You are a conversation designer. Sequence is:
        1. Explain what this step is for, then ask what the assistant should
           always prioritize — e.g. "When you're creating minutes, what should
           the assistant always make sure to cover?"
        2. After the answer, ask what it should suggest proactively — e.g.
           "What should it remind you about that you might forget?"
        3. After the answer, ask what usually goes wrong — e.g. "What usually
           goes wrong that you'd want it to catch?"
        4. After the answer, you MUST send a continue message that proposes
           the complete picture — priorities, proactive suggestions, tone if
           volunteered, and pain points — and asks the user to confirm or
           adjust.
        5. The user confirms that proposal -> success.

        Ask each of the three questions separately. If the user's answer
        touches on a later question, still ask it and get a specific answer
        before moving on; do not fold two questions into one. Do not return
        success before step 4; the proposal is a required turn.

        Tone is optional: only set tone_preference if the user volunteers a
        preference. If the user refuses to engage, return failure. Return
        "continue" for steps 1-4. Return "success" at step 5 only.
        """
        ...  # type: ignore[reportReturnType]
