"""Domain specification models for workflow components.

ComponentDomainSpec captures the domain-level understanding of a business
artifact — what it means in the user's world, what fields it has, and what
quality makes it excellent. This is pure domain language, free of any
Python or Pydantic implementation details.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from chat_workflow import atomic_workflow

from .component_responsibilities import ComponentResponsibilities


class ComponentDomainField(BaseModel):
    """A single field within a domain artifact.

    Describes what this field means in the user's world, using their
    terminology — not Python types or data structure names.
    """

    name: str
    """Field name in domain terms (e.g. "action owner", "decision")."""

    domain_description: str
    """What this field means in the user's world.

    Example: "The person responsible for completing this action item"
    """

    field_type_hint: str
    """Domain type hint describing the kind of value expected.

    Example: "person name", "date", "multiple action items"
    """


class ComponentDomainSpec(BaseModel):
    """Domain-level specification for a business artifact.

    Captures what the artifact represents in the user's domain, the
    fields it contains, holistic quality criteria, and who owns it.
    The name must match the corresponding ComponentResponsibilities.name.
    """

    name: str
    """Matches ComponentResponsibilities.name — the artifact name."""

    description: str
    """What this artifact represents in the user's domain.

    Example: "Structured meeting minutes that capture what happened"
    """

    fields: list[ComponentDomainField]
    """Fields that make up this artifact, described in domain terms."""

    what_good_looks_like: list[str]
    """Holistic quality criteria for the artifact as a whole.

    These are NOT field-specific business rules. They describe what
    makes the overall artifact excellent.

    Example: "Attendees can immediately understand decisions made"
    """

    expert_role: str
    """Who owns this domain — the role responsible for this artifact."""

    @atomic_workflow
    @classmethod
    def explore(
        cls,
        responsibilities: Annotated[
            ComponentResponsibilities,
            "What this component is responsible for — its name, purpose, scope, and any incidental notes",
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ComponentDomainSpec:
        """You are a domain analyst helping someone define what a business artifact
        looks like in their world.

        The user has already described what this artifact is responsible for — its
        purpose, scope, and boundaries. Your job is to help them flesh out the
        details: what information it should capture, what makes it excellent, and
        who owns it.

        CRITICAL — Response format rules:
        - When you need to ask the user a question or propose ideas for discussion,
          use "continue" intent with a message only — do NOT include a result field.
        - Only use "success" intent when the user has confirmed the complete domain
          spec and you are ready to return the final result.
        - Never include a result with "continue" intent. Never include a message with
          "success" intent.

        This isn't a form to fill out. You're an expert who has seen many similar
        artifacts in different organizations. When the user describes their needs,
        propose a complete picture back to them rather than asking about each piece
        one at a time.

        Start by proposing the artifact fields you think it should have based on
        what you already know about its purpose and scope. For example: "From what
        you've described, this meeting minutes artifact would capture the meeting
        date and attendees, key decisions made, action items with owners and due
        dates, and any open questions that need follow-up. Does that cover what you
        had in mind?"

        Once the user confirms or adjusts the fields, ask about what makes the
        overall artifact excellent — not field-level rules, but holistic quality.
        For example: "What would make these minutes excellent rather than just
        adequate? For instance, should attendees be able to immediately understand
        what was decided without reading every word? Should someone who missed the
        meeting be able to catch up in two minutes?"

        Then ask who the expert is — the role that owns this artifact and is
        responsible for its quality. For example: "Who typically owns the quality
        of these minutes? Is there a specific role — like a meeting coordinator or
        a team lead — who reviews and approves them?"

        Stay entirely in the user's domain language. Talk about what the artifact
        means in their work, what information it carries, what makes it good. Never
        mention data structures, fields, types, or any technical implementation
        concepts.

        Do not re-ask or re-confirm what was already settled. If the user confirms
        your proposal, move on to the next topic. If they correct something, update
        your understanding and propose the revised picture — don't ask a follow-up
        question about each correction separately.

        Never put fabricated values in the final output. Only include what the user
        has confirmed. But you can propose ideas in conversation — that's how you
        help them think through what they need.
        """
        ...  # type: ignore[reportReturnType]
