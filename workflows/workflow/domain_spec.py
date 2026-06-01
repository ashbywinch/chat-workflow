"""Domain specification models for workflow components.

ComponentDomainSpec captures the domain-level understanding of a business
artifact — what it means in the user's world, what fields it has, and what
quality makes it excellent. This is pure domain language, free of any
Python or Pydantic implementation details.
"""

from pydantic import BaseModel


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
