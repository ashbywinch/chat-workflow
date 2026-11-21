"""ComponentStructure model — Pydantic-level structural design of a generated component.

ComponentStructure captures the Pydantic model structure that will be generated
for a workflow component. It describes fields, validators, and imports needed
to produce the final Python code — without any LLM or conversation concepts.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.confirmation import confirm_draft
from chat_workflow.rendering import RenderContext, plain_type

from .domain_spec import ComponentDomainSpec


class StructField(BaseModel):
    """A single field in the generated Pydantic model.

    Describes the field's name, Python type expression, and any Pydantic
    Field() kwargs that should be applied. The type_expr and field_def_kwargs
    use string values because they will be serialized into generated Python code.
    """

    name: str = Field(
        ...,
        description="Field name (e.g. 'action_owner', 'description')",
        min_length=1,
    )
    type_expr: str = Field(
        ...,
        description=("Python type expression as a string (e.g. 'str', 'list[str]', 'list[ActionItem]')"),
        min_length=1,
    )
    field_def_kwargs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Pydantic Field kwargs as string key-value pairs (e.g. {'description': '...', 'min_length': '1'})"
        ),
    )
    field_type_hint: str = Field(
        "",
        description=(
            "The user's words for this field's type, copied from the domain "
            "spec (for example 'date' or 'list of person names'). Empty means "
            "the type expression is described instead."
        ),
    )
    field_description: str = Field(
        "",
        description=(
            "What the field means in the user's world, copied from the domain "
            "spec's field description."
        ),
    )


_TYPE_IN_RULE = re.compile(
    r"\b(str|int|float|bool|dict|datetime)\b"
    r"|\b(list|set|tuple|dict|Optional)\["
    r"|\|\s*None\b"
)


class StructValidator(BaseModel):
    """A validation rule for the generated Pydantic model.

    Captures a business rule or quality criterion that should be enforced
    as a @model_validator or @field_validator in the generated code.
    """

    rule: str = Field(
        ...,
        description=("Validation rule text (e.g. 'description must not exceed 3 sentences')"),
        min_length=1,
    )
    domain_origin: str = Field(
        ...,
        description=("The 'what good looks like' statement or business rule this validation came from"),
        min_length=1,
    )

    @field_validator("rule")
    @classmethod
    def _the_rule_is_prose(cls, value: str) -> str:
        """A rule is the user's words; a type expression is not a rule."""
        if _TYPE_IN_RULE.search(value):
            raise ValueError(
                "A rule is prose in the user's words, not a type expression: "
                "say 'a description', 'a list of names', 'an optional date', "
                "never 'str', 'list[...]', or '| None'."
            )
        return value


class ComponentStructure(BaseModel):
    """Pydantic-level structural design for a generated component.

    Represents the complete Pydantic model structure that will be code-generated:
    the artifact description, base class, fields, validators, and extra imports.
    This is a pure data model — no LLM functionality.
    """

    model_config = ConfigDict(title="the shape")

    description: str = Field(
        ...,
        description="Refined artifact description for the generated model",
        min_length=1,
    )
    base_class: str = Field(
        "LLMValidated",
        description=(
            "Base class for the generated model: 'LLMValidated' (required — "
            "the component enforces its own domain rules)"
        ),
    )
    fields: list[StructField] = Field(
        default_factory=list,
        description="Fields that make up the generated Pydantic model",
    )
    model_validators: list[StructValidator] = Field(
        default_factory=list,
        description="Validation rules for the generated model",
    )
    extra_imports: list[str] = Field(
        default_factory=list,
        description="Additional Python imports needed by the generated code",
    )

    def render(self, context: RenderContext) -> str:
        """Say the shape in the domain's words, carried by each field."""
        lines = []
        for field in self.fields:
            description = field.field_description or field.field_def_kwargs.get(
                "description"
            )
            hint = field.field_type_hint or plain_type(field.type_expr)
            line = f"{field.name.replace('_', ' ')}: {description or hint}"
            if description:
                line += f" ({hint})"
            lines.append(line)
        if self.model_validators:
            lines.extend(f"Rule: {validator.rule}" for validator in self.model_validators)
        return "\n".join(lines)

    @composite_workflow_step
    @classmethod
    def design(
        cls,
        domain_spec: Annotated[
            ComponentDomainSpec,
            (
                "The domain specification — what the artifact means in the user's"
                " world, its fields, and what makes it excellent"
            ),
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]
    ) -> ComponentStructure:
        """Derive the structure from the domain spec, then confirm it."""
        draft = _draft_structure(domain_spec=domain_spec, session=session)
        session.propose(draft)
        return confirm_draft(
            session=session,
            draft=draft,
            step=_confirm_structure,
            max_turns=min(max_turns, 4),
        )


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_structure(
    domain_spec: Annotated[
        ComponentDomainSpec,
        "What the artifact means in the user's world, its fields, and what makes it excellent",
    ],
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> ComponentStructure:
    """Derive the structure from the domain spec.

The domain spec already lists the fields and the type hints. Derive the complete ComponentStructure yourself; do not
ask the user anything. For each field, copy its `field_type_hint` and `field_description` from the domain spec into the
structure field; those are the user's own words for the type and its meaning. Then choose the Python type and any
structurally enforceable rule. Translate
the business
language into the technical structure silently. For dates, use a str with an ISO format or a pre-validator; never a
date/datetime field. Include a validator for every rule a field can enforce structurally; return an empty list only
when nothing is structurally enforceable. Do not define a second independent artifact."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step
def _confirm_structure(
    draft: Annotated[
        ComponentStructure,
        "The complete structure already shown to the user",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> ComponentStructure:
    """
The structure was shown with one confirm-or-correct question, and the user is
answering it. Do not show it again and do not repeat that question.

If the user confirms, return the draft. If they correct a type or a rule, update it and return the corrected
structure; the framework shows the change. Ask only about a gap the draft cannot fill; do not re-present the whole
structure or ask the user to choose code types. Say the types in business words — "text", "a list of names", "an
optional date" — never "str", "list[str]", or "| None".
    
    """
    ...  # type: ignore[reportReturnType]
