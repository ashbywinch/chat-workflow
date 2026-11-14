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
from chat_workflow.annotations import UserLabel
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

    name: Annotated[
        str,
        UserLabel("what you call it", "name"),
    ] = Field(
        ...,
        description="Matches ComponentSpec.name — the artifact name",
        min_length=1,
    )

    description: Annotated[
        str,
        UserLabel("what it is", "description"),
    ] = Field(
        ...,
        description="What this artifact represents in the user's domain",
        min_length=1,
    )

    fields: Annotated[
        list[ComponentDomainField],
        UserLabel("the parts of it", "fields"),
    ] = Field(
        ...,
        description="Fields that make up this artifact, described in domain terms",
        min_length=1,
    )

    what_good_looks_like: Annotated[
        list[str],
        UserLabel("what makes it good", "what good looks like"),
    ] = Field(
        ...,
        description="Holistic quality criteria for the artifact as a whole — what makes it excellent",
        min_length=1,
    )

    expert_role: Annotated[
        str,
        UserLabel("the expertise I should bring", "expert role"),
    ] = Field(
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
        """You are a domain analyst. Propose the artifact's fields from its
        responsibilities first, and ask the user to confirm them. Propose
        once. If they already described what good looks like in their
        confirmation, acknowledge it and ask about the assistant's expertise
        separately; otherwise ask about quality once. Then return "success".

        Only after the user has responded to the field list, confirm the
        expertise in plain language: "Would it help if I focused on structuring
        the highlights?" The expertise is your choice, inferred from the
        artifact; never ask the user to choose a job title or ask "what title
        should I act as?" For meeting minutes, the owner is the meeting
        organizer or chair; a scribe records the notes, which is neither the
        owner nor the assistant's expertise. If the user offers a sample or
        moves to the next practical step, accept and follow it; do not divert
        to the role question. A field list derived from the responsibilities
        is a draft for approval, not a hypothesis dump. Return at least one
        field and one quality criterion.

        Use the scope to decide what belongs in the artifact: it is this
        component's own work. Do not add fields for work the scope says
        belongs to another component or is out of scope.

        After the user answers a question, move on to the next topic.
        Return "success" with what you have rather than pressing for a
        complete answer.
        """
        ...  # type: ignore[reportReturnType]
