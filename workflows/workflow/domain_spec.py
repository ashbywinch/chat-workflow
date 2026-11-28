"""Domain specification models for workflow components.

ComponentDomainSpec captures the domain-level understanding of a business
artifact — what it means in the user's world, what fields it has, and what
quality makes it excellent. This is pure domain language, free of any
Python or Pydantic implementation details.
"""

from __future__ import annotations

import re
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.confirmation import confirm_draft
from chat_workflow.mixins import LLMValidated
from chat_workflow.rendering import RenderContext

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

    model_config = ConfigDict(title="the artifact")

    _validation_rules: ClassVar[list[str]] = [
        "The expert_role is the agent's domain expertise — the role the agent "
        "acts as when helping with this artifact. It may equal the owner_role "
        "or name a specialist helping the owner. It is not a generic label "
        "('Reviewer', 'Expert', 'Analyst', 'Assistant', 'Helper') and not the "
        "person who supplies the input unless they are that specialist.",
        # Rules about rules: what_good_looks_like is the standard the component
        # will be judged against, so each entry must be something a reader can
        # evaluate without guessing the author's intent.
        "Each entry in what_good_looks_like is one requirement. Two attributes "
        "of the same thing ('an owner and a due date') are one requirement; two "
        "different outcomes are two entries.",
        "A quality word carries its standard. 'Concise', 'readable', 'polite' "
        "are fine: their standard is intrinsic and shared. 'Complete' (with "
        "respect to what?), 'sufficient' (for what?), 'appropriate' (to what?), "
        "'specific' (about what?) must name their reference or be replaced by "
        "the parts that must be present. Non-deterministic is not ambiguous: do "
        "not demand a mechanical test for 'concise'.",
        "An entry does not encode a value from one example (a place name, a "
        "date, a person's name) as a requirement.",
        "A requirement a program could decide exactly — a field is present, a "
        "date has a format, an owner is not empty — belongs in the fields and "
        "validators, not in what_good_looks_like.",
    ]

    name: Annotated[
        str,
        UserLabel(phrase="what you call it", aliases="name", guard=False),
    ] = Field(
        ...,
        description="Matches ComponentSpec.name — the artifact name",
        min_length=1,
    )

    description: Annotated[
        str,
        UserLabel(phrase="what it is", aliases="description", guard=False),
    ] = Field(
        ...,
        description="What this artifact represents in the user's domain",
        min_length=1,
    )

    fields: Annotated[
        list[ComponentDomainField],
        UserLabel(phrase="the parts of it", aliases="fields"),
    ] = Field(
        ...,
        description="Fields that make up this artifact, described in domain terms",
        min_length=1,
    )

    what_good_looks_like: Annotated[
        list[str],
        UserLabel(phrase="what makes it good", aliases="what good looks like"),
    ] = Field(
        ...,
        description="Holistic quality criteria for the artifact as a whole — what makes it excellent",
        min_length=1,
    )

    expert_role: Annotated[
        str,
        UserLabel(phrase="the expertise I should bring", aliases="expert role"),
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

    def render(self, context: RenderContext) -> str:
        """Say what the artifact is and what makes it good."""
        fields = "; ".join(
            f"{field.name} — {field.domain_description}" for field in self.fields
        )
        lines = [
            f"{self.name} — {self.description}",
            f"Fields: {fields}",
            "What makes it good: " + "; ".join(self.what_good_looks_like),
            f"The role is {self.expert_role}.",
        ]
        return "\n".join(lines)

    @field_validator("name")
    @classmethod
    def _name_is_a_component_name(cls, value: str) -> str:
        """A component name is a Python class name, not a display phrase.

        The model is told to keep the agreed name, and usually does. When it
        writes "Meeting Minutes" for "MeetingMinutes", the contract still
        holds: the name is the one ComponentSpec carries, and Component uses
        it as the class name.
        """
        parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
        cleaned = "".join(part[:1].upper() + part[1:] for part in parts)
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

    @composite_workflow_step
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
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]  # the step decorator injects the session
    ) -> ComponentDomainSpec:
        """Derive the artifact's domain spec, then confirm it."""
        draft = _draft_domain_spec(
            responsibilities=responsibilities,
            deliverables=deliverables,
            session=session,
        )
        session.propose(draft)
        return confirm_draft(
            session=session,
            draft=draft,
            step=_confirm_domain_spec,
        )


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_domain_spec(
    responsibilities: Annotated[
        ComponentSpec,
        "What this part of the work is responsible for, in the user's words",
    ],
    deliverables: Annotated[
        list[Deliverable] | None,
        "What the user makes, with the check they gave for it",
    ] = None,
) -> ComponentDomainSpec:
    """Derive the artifact's domain spec from the agreed responsibilities.

The responsibilities and any deliverables are established. Derive the fields, the quality checks, and the expertise
yourself; do not ask the user anything. Keep `responsibilities.name` exactly as given: do not reformat, split, or
expand it. Return the complete spec."""
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body


@atomic_workflow_step
def _confirm_domain_spec(
    draft: Annotated[
        ComponentDomainSpec,
        "The complete domain spec already shown to the user",
    ],
) -> ComponentDomainSpec:
    """
The domain spec was shown with one confirm-or-correct question, and the user is
answering it. Do not show it again and do not repeat that question.

If the user confirms, return the draft. If they correct or add a field or check — including a section, a table, or a
length limit — update the fields or what_good_looks_like and return the corrected spec; the framework shows the change.
Ask only about a gap the draft cannot fill; do not re-present the whole spec and do not ask what to call the artifact.
If a quality check is ambiguous — a word whose reference is unstated — ask which reading the user means and update it;
do not choose one for them.
If they move to a different part of the work — how the artifact gets made, who does what, what happens next — say in one
sentence that this step defines the artifact itself and that part comes next, then finish. Do not restate a change you
already reported in an earlier turn; say what changed only in the turn where it changes.
    
    """
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body


from .models.deliverable import Deliverable  # noqa: E402 — imported after the class to break the model cycle
