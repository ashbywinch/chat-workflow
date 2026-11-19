"""Domain specification models for workflow components.

ComponentDomainSpec captures the domain-level understanding of a business
artifact — what it means in the user's world, what fields it has, and what
quality makes it excellent. This is pure domain language, free of any
Python or Pydantic implementation details.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, field_validator

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
        "('Reviewer', 'Expert', 'Analyst', 'Assistant', 'Helper') and not the "
        "person who supplies the input unless they are that specialist.",
    ]

    name: Annotated[
        str,
        UserLabel("what you call it", "name", guard=False),
    ] = Field(
        ...,
        description="Matches ComponentSpec.name — the artifact name",
        min_length=1,
    )

    description: Annotated[
        str,
        UserLabel("what it is", "description", guard=False),
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
            "specialist helping the owner. Name a specific domain role, not "
            "'Assistant' or 'Helper'."
        ),
        min_length=1,
    )

    @field_validator("name")
    @classmethod
    def _name_is_a_component_name(cls, value: str) -> str:
        """A component name is a Python class name, not a display phrase.

        The model is told to keep the agreed name, and usually does. When it
        writes "Meeting Minutes" for "MeetingMinutes", the contract still
        holds: the name is the one ComponentSpec carries, and Component uses
        it as the class name.
        """
        cleaned = "".join(ch for ch in value if ch.isalnum() or ch == "_")
        return cleaned or value

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
            "What this part of the work is responsible for, in the user's words",
        ],
        deliverables: Annotated[
            list[Deliverable] | None,
            "What the user makes, with the check they gave for it",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ComponentDomainSpec:
        """Shape what the thing contains with the user.

    The responsibilities are already agreed, so do not ask the user to confirm the shape again. Keep
    `responsibilities.name` exactly as given: do not reformat, split, or expand it. State the parts the thing needs, in
    their words, and move on; do not turn the shape into a second confirmation. Open by proposing the fields from the
    responsibilities and any deliverables; do not ask what to call the thing or ask the user to list its parts from
    scratch.

    If `deliverables` is given, the success criteria it carries is the check the user already gave: use it as the
    quality checks and do not ask for quality again. If there are no deliverables, propose the checks from your domain
    knowledge and ask the user to confirm or add.

    Infer the expertise you need and confirm it in one sentence. Ask one question at a time and do not interrogate
    them.

    Return the result when they confirm."""
        ...  # type: ignore[reportReturnType]

from .models.deliverable import Deliverable  # noqa: E402
