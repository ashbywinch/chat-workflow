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

    conversation_design_pattern: str | None = None
    """Optional conversation pattern hint for the code generator.

    ``None`` or ``"exploration-then-structure"`` for entry-point workflows
    (warm open, explore, then propose structure).
    ``"efficient-fill"`` for sub-conversations (no greeting, propose and
    confirm fields efficiently).
    """

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
        """You are a domain analyst. The user described what their artifact
        does. Propose fields, ask about quality, then who owns it.

        First message: propose fields. "Meeting minutes would have date,
        attendees, decisions, action items — right?"

        Second message: after they confirm, ask about quality. "What makes
        these excellent?" If they mention the owner in their answer, use
        that instead of asking again.

        Third message: after quality, return with "success". Do NOT ask
        any question more than once. When the user answers, respond and
        move to the next topic.
        """
        ...  # type: ignore[reportReturnType]
