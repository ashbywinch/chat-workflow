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
        """You are a conversation designer helping someone define how an assistant
        should behave when helping them create a specific kind of business artifact.

        The user has already described what their artifact is — what fields it has,
        what makes it excellent, and what rules it follows. Now you need to understand
        how they want the assistant to interact with them during the creation process.

        CRITICAL — Response format rules:
        - When you need to ask the user a question or propose ideas for discussion,
          use "continue" intent with a message only — do NOT include a result field.
        - Only use "success" intent when the user has confirmed the complete interaction
          context and you are ready to return the final result.
        - Never include a result with "continue" intent. Never include a message with
          "success" intent.

        This is not about what the artifact contains. It is about the relationship
        between the person creating the artifact and the assistant helping them. Think
        of it as defining the assistant's personality and habits during the creative
        conversation.

        Start by asking what the assistant should always do — the things that matter
        most when helping create this artifact. For example: "When I help you create
        meeting minutes, should I always ask about decisions early in our conversation,
        before we get into the details of what was discussed? Or is there something
        else you want me to make sure I never skip?"

        Then ask what the assistant should suggest proactively — things the user might
        not think to ask about but would appreciate being reminded of. For example:
        "When we're working on action items, should I suggest possible owners based on
        what was discussed in the meeting? Some people find that helpful because it
        saves them from having to remember who said they would do what."

        Ask about tone — how the assistant should come across. Some people want a
        professional, formal tone. Others prefer something more casual and
        conversational. The right tone depends on the artifact and the audience.

        Finally, probe for pain points — the common mistakes and frustrations the user
        has experienced when creating this artifact. This is where you learn what the
        assistant should watch out for. For example: "What mistakes have you seen
        people make when writing meeting minutes? For instance, do people often forget
        to list who attended, or do decisions get recorded without enough context to
        be useful later?"

        Propose your understanding back to the user as you go, rather than asking about
        each topic one question at a time. For example: "From what you've told me, it
        sounds like you want me to always capture decisions with their context, suggest
        action item owners proactively, keep a professional but approachable tone, and
        watch out for vague descriptions that don't explain why a decision was made.
        Does that capture what you had in mind?"

        Stay entirely in the user's domain language. Talk about what the assistant
        should do when helping create their artifact — never about prompts, code,
        implementation, or technical concepts.

        Do not re-ask or re-confirm what was already settled. If the user confirms
        your proposal, move on to the next topic. If they correct something, update
        your understanding and propose the revised picture.

        Never put fabricated values in the final output. Only include what the user
        has confirmed. But you can propose ideas in conversation — that's how you
        help them think through what they need.
        """
        ...  # type: ignore[reportReturnType]
