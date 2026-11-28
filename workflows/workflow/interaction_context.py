"""ComponentInteractionContext — how the assistant should work with the user."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.confirmation import confirm_draft
from chat_workflow.rendering import RenderContext

from .domain_spec import ComponentDomainSpec
from .structure import ComponentStructure


class ComponentInteractionContext(BaseModel):
    """What the assistant should prioritize, suggest, and watch for.

    This is preference data from the user. The questions are fixed by this
    schema, so the framework asks them; the model derives the picture from the
    answer, the renderer shows it, and a confirmation step settles it.
    """

    model_config = ConfigDict(title="how I should work with you")

    must_prioritize: Annotated[
        list[str],
        UserLabel(phrase="what I should always cover", aliases="must prioritize"),
    ] = Field(
        ...,
        description=(
            "What the assistant should always address during the conversation, "
            "e.g. 'Always ask about decisions early in the conversation'"
        ),
    )
    auto_suggest: Annotated[
        list[str],
        UserLabel(phrase="what I should suggest", aliases="auto suggest"),
    ] = Field(
        ...,
        description=(
            "What the assistant should suggest proactively to the user, "
            "e.g. 'Suggest action item owners based on the topic discussed'"
        ),
    )
    tone_preference: Annotated[
        str | None,
        UserLabel(phrase="how I should sound", aliases="tone preference"),
    ] = Field(
        default=None,
        description=(
            "Tone guidance for the assistant's communication style, "
            "e.g. 'Professional but friendly'. None means no preference."
        ),
    )
    user_pain_points: Annotated[
        list[str],
        UserLabel(phrase="what usually goes wrong", aliases="user pain points"),
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
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]  # the step decorator injects the session
    ) -> ComponentInteractionContext:
        """Ask the three fixed questions, show the derived picture, confirm it."""
        answer = session.io.prompt(
            "When I help you create this, what should I always cover? "
            "What would you like me to suggest proactively? "
            "What usually goes wrong that I should catch?"
        )
        draft = _draft_interaction_context(
            domain_spec=domain_spec,
            structure=structure,
            answer=answer,
            session=session,
        )
        session.propose(draft)
        return confirm_draft(
            session=session,
            draft=draft,
            step=_confirm_interaction_context,
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
    answer: Annotated[str, "The user's answer to the three questions"],
) -> ComponentInteractionContext:
    """Derive the interaction picture from the user's answer.

The three questions are what the assistant should always cover, what it should suggest proactively, and what usually
goes wrong; the user's answer is in `answer`. Derive all four parts from it, using the domain spec and structure for
the artifact. If the answer leaves one part empty, propose the usual answer for this kind of work. Do not ask the user
anything. Leave the tone empty unless the user gave one. Return the complete context."""
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body


@atomic_workflow_step
def _confirm_interaction_context(
    draft: Annotated[
        ComponentInteractionContext,
        "The complete interaction proposal already shown to the user",
    ],
) -> ComponentInteractionContext:
    """
The proposal was shown with one confirm-or-correct question, and the user is
answering it. Do not show it again and do not repeat that question.

If the user confirms, return the draft. If they correct a point, update it and return the corrected context; the
framework shows the change. Ask only about a point they left unanswered. Do not re-present the whole proposal or ask
what to call the artifact.
    
    """
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body
