"""ComponentStructure model — Pydantic-level structural design of a generated component.

ComponentStructure captures the Pydantic model structure that will be generated
for a workflow component. It describes fields, validators, and imports needed
to produce the final Python code — without any LLM or conversation concepts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step

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


class ComponentStructure(BaseModel):
    """Pydantic-level structural design for a generated component.

    Represents the complete Pydantic model structure that will be code-generated:
    the artifact description, base class, fields, validators, and extra imports.
    This is a pure data model — no LLM functionality.
    """

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
        session.presented_hints = {
            field.name: field.field_type_hint for field in domain_spec.fields
        }
        session.present(result=draft, text=_format_structure(draft, domain_spec))
        return _confirm_structure(
            draft=draft, session=session, max_turns=min(max_turns, 4)
        )


def _plain_type(type_expr: str) -> str:
    """Say a type in business words, not Python syntax."""
    text = type_expr.strip()
    text = text.replace("| None", "").strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1]
    if text == "str":
        plain = "text"
    elif text in {"int", "float"}:
        plain = "a number"
    elif text == "bool":
        plain = "yes or no"
    elif text.startswith("list[") and text.endswith("]"):
        plain = "a list of " + _plain_type(text[len("list[") : -1])
    elif text.startswith("dict[") or text == "dict":
        plain = "a set of values"
    else:
        plain = text.replace("_", " ").casefold()
    return plain


def _format_structure(
    draft: ComponentStructure, domain_spec: ComponentDomainSpec
) -> str:
    """Present the derived structure in the user's words."""
    hints = {field.name: field.field_type_hint for field in domain_spec.fields}
    descriptions = {
        field.name: field.domain_description for field in domain_spec.fields
    }
    lines = ["Now I'll turn what we agreed into a concrete shape.", "The parts:"]
    for field in draft.fields:
        hint = hints.get(field.name) or _plain_type(field.type_expr)
        description = descriptions.get(field.name)
        line = f"- {field.name.replace('_', ' ')}: {hint}"
        if description:
            line += f" — {description}"
        lines.append(line)
    if draft.model_validators:
        lines.append("The rules:")
        lines.extend(f"- {validator.rule}" for validator in draft.model_validators)
    lines.append("Does this capture it, or would you change a part?")
    return "\n".join(lines)


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
ask the user anything. For each field, choose the type and any structurally enforceable rule. Translate the business
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
    """The structure has been shown. Ask one confirm-or-correct question.

If the user confirms, return the draft. If they correct a type or a rule, say in one sentence what you changed,
update it, and return the corrected structure. Ask only about a gap the draft cannot fill; do not re-present the whole
structure or ask the user to choose code types. Say the types in business words — "text", "a list of names",
"an optional date" — never "str", "list[str]", or "| None"."""
    ...  # type: ignore[reportReturnType]
