"""Domain specification models for workflow components.

ComponentDomainSpec captures the domain-level understanding of a business
artifact — what it means in the user's world, what fields it has, and what
quality makes it excellent. This is pure domain language, free of any
Python or Pydantic implementation details.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated

from .component_spec import ComponentSpec


class ComponentDomainField(BaseModel):
    """A single field within a domain artifact.

    Describes what this field means in the user's world, using their
    terminology — not Python types or data structure names.
    """

    name: str = Field(
        ...,
        description='Field name in domain terms (e.g. "action owner", "decision")',
        min_length=1,
    )

    domain_description: str = Field(
        ...,
        description="What this field means in the user's world",
        min_length=1,
    )

    field_type_hint: str = Field(
        ...,
        description='Domain type hint (e.g. "person name", "date", "multiple action items")',
        min_length=1,
    )


class ComponentDomainSpec(LLMValidated):
    """Domain-level specification for a business artifact.

    Captures what the artifact represents in the user's domain, the
    fields it contains, holistic quality criteria, and who owns it.
    The name must match the corresponding ComponentSpec.name.
    """

    _validation_rules: ClassVar[list[str]] = [
        "The expert_role is the agent's domain expertise — the role the agent "
        "acts as when helping with this artifact. It may equal the owner_role "
        "or name a specialist helping the owner. It is not a generic label "
        "('Reviewer', 'Expert', 'Analyst') and not the person who supplies the "
        "input unless they are that specialist.",
    ]

    name: str = Field(
        ...,
        description="Matches ComponentSpec.name — the artifact name",
        min_length=1,
    )

    description: str = Field(
        ...,
        description="What this artifact represents in the user's domain",
        min_length=1,
    )

    fields: list[ComponentDomainField] = Field(
        ...,
        description="Fields that make up this artifact, described in domain terms",
        min_length=1,
    )

    what_good_looks_like: list[str] = Field(
        ...,
        description="Holistic quality criteria for the artifact as a whole — what makes it excellent",
        min_length=1,
    )

    expert_role: str = Field(
        ...,
        description=(
            "The agent's domain expertise — the role the agent acts as when "
            "helping with this artifact. It may equal the owner_role or be a "
            "specialist helping the owner."
        ),
        min_length=1,
    )

    conversation_design_pattern: str | None = Field(
        default=None,
        description=(
            'Optional conversation pattern: "exploration-then-structure" for an '
            'entry-point workflow (warm open, explore, then propose structure) or '
            '"efficient-fill" for a sub-conversation (no greeting; propose and '
            "confirm efficiently). None means the default pattern."
        ),
    )

    @atomic_workflow_step
    @classmethod
    def explore(
        cls,
        responsibilities: Annotated[
            ComponentSpec,
            "What this component is responsible for — its name, purpose, scope, and any incidental notes",
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ComponentDomainSpec:
        """You are a domain analyst. Propose the artifact's fields from
        its responsibilities. Propose once. The user confirms (maybe
        with elaboration). If they already described what good looks
        like in their confirmation, acknowledge and return "success".
        Otherwise ask about quality once. Then return "success".

        The owner role is already in `responsibilities.owner_role`. Propose
        the agent's role — "the agent acts as …" — the expertise it needs for
        this artifact. It may be the same as the owner role or a specialist
        helping them. Confirm it with the user. For meeting minutes, the agent
        acts as a "Meeting Minutes Specialist"; the owner is the "Meeting
        Organizer" or "Meeting Chair". A "Scribe" records the notes; it is
        neither the owner nor the agent's expertise. A field list derived
        from the responsibilities is a draft for approval, not a hypothesis
        dump. Return at least one field and one quality criterion.

        Use the scope to decide what belongs in the artifact: it is this
        component's own work. Do not add fields for work the scope says
        belongs to another component or is out of scope.

        After the user answers a question, move on to the next topic.
        Return "success" with what you have rather than pressing for a
        complete answer.
        """
        ...  # type: ignore[reportReturnType]
