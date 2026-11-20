"""ComponentInteractionContext model — domain considerations for assistant interaction."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.rendering import RenderContext

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

    def render(self, context: RenderContext) -> str:
        """Say how the assistant proposes to work with the user."""
        lines: list[str] = []
        if self.must_prioritize:
            lines.append("Always cover: " + "; ".join(self.must_prioritize))
        if self.auto_suggest:
            lines.append("Suggest: " + "; ".join(self.auto_suggest))
        if self.tone_preference:
            lines.append(f"Tone: {self.tone_preference}")
        if self.user_pain_points:
            lines.append("Watch for: " + "; ".join(self.user_pain_points))
        return "\n".join(lines)

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
        session.propose(draft)
        return _confirm_interaction_context(
            draft=draft, session=session, max_turns=min(max_turns, 4)
        )


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
