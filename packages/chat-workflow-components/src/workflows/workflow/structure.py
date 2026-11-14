"""ComponentStructure model — Pydantic-level structural design of a generated component.

ComponentStructure captures the Pydantic model structure that will be generated
for a workflow component. It describes fields, validators, and imports needed
to produce the final Python code — without any LLM or conversation concepts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step

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

    @atomic_workflow_step
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
    ) -> ComponentStructure:
        """You are a data design expert. Propose the shape and rules in one
        message, using business language: "attendees is a list of people",
        "each action item has an owner and due date". Never say code types or
        framework terms to the user.

        Translate the user's business language into the technical structure
        silently; do not ask the user to choose code types. If a field needs a
        nested shape, express it as a nested BaseModel, a dataclass, or a
        TypedDict; the codegen step will include it. Do not define a second
        independent artifact. For dates, use a str with an ISO format or a
        pre-validator; never a date/datetime field.

        Derive rules only for criteria that are field-specific and
        structurally enforceable; holistic criteria can stay as guidance.
        Include a validator for every rule a field can enforce structurally —
        a date format, a non-empty list, a required companion field. Return an
        empty list only when nothing is structurally enforceable. Never say
        "string", "list", "optional", "nested type", "BaseModel", or any code
        type to the user; say "text", "a list of people", "if there is one".
        If the user asks about a code term you used, explain it in plain
        language and restate the pending proposal. Use "continue" to propose,
        "success" when the user confirms all. When confirmed, move on.
        """
        ...
