"""ComponentInteractionContext model — domain considerations for assistant interaction."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
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

    @composite_workflow_step
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
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]
    ) -> ComponentInteractionContext:
        """Derive how the assistant should interact, then confirm it."""
        draft = _draft_interaction_context(
            domain_spec=domain_spec,
            structure=structure,
            session=session,
        )
        session.present(result=draft, text=_format_interaction_context(draft))
        return _confirm_interaction_context(
            draft=draft, session=session, max_turns=min(max_turns, 4)
        )


def _format_interaction_context(draft: ComponentInteractionContext) -> str:
    """Present the interaction proposal in the user's words."""
    lines = ["Now I'll propose how I should work with you.", "", "Here's how I'd work with you:"]
    if draft.must_prioritize:
        lines.append("- I'll always cover: " + "; ".join(draft.must_prioritize))
    if draft.auto_suggest:
        lines.append("- I'll suggest: " + "; ".join(draft.auto_suggest))
    if draft.tone_preference:
        lines.append("- Tone: " + draft.tone_preference)
    if draft.user_pain_points:
        lines.append("- I'll watch for: " + "; ".join(draft.user_pain_points))
    lines.append("")
    lines.append(
        "What would you add to what I should always prioritize? What would you "
        "like me to suggest proactively? What usually goes wrong that I should "
        "catch?"
    )
    return "\n".join(lines)


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_interaction_context(
    domain_spec: Annotated[
        ComponentDomainSpec,
        "What the artifact means in the user's world and what makes it excellent",
    ],
    structure: Annotated[
        ComponentStructure,
        "What fields and rules the artifact follows",
    ],
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> ComponentInteractionContext:
    """Derive how the assistant should interact with the user.

The domain spec and structure are established. Derive all four parts yourself; do not ask the user anything:
1. What the assistant should always cover.
2. What it should suggest proactively.
3. The tone, only if the user volunteered one; otherwise leave it empty.
4. What usually goes wrong that the assistant should catch.
Never ask what to call the artifact; the domain spec names it. Return the complete context."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step
def _confirm_interaction_context(
    draft: Annotated[
        ComponentInteractionContext,
        "The complete interaction proposal already shown to the user",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ComponentInteractionContext:
    """The proposal and the three questions have already been shown. Use the user's answers.

Ask only about a point they left unanswered. If they correct a value, say in one sentence what you changed, update it,
and return the corrected context. If they confirm, return the draft. Do not re-present the whole proposal, repeat the
three questions, or ask what to call the artifact."""
    ...  # type: ignore[reportReturnType]
