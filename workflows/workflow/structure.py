"""ComponentStructure model — Pydantic-level structural design of a generated component.

ComponentStructure captures the Pydantic model structure that will be generated
for a workflow component. It describes fields, validators, and imports needed
to produce the final Python code — without any LLM or conversation concepts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

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
        "BaseModel",
        description=("Base class for the generated model: 'BaseModel' (default) or 'LLMValidated'"),
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

    @atomic_workflow
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
        """You are a data design expert. The user described their artifact in
        domain terms; now translate that into concrete structural decisions.

        Propose types and constraints in a single picture, then confirm.
        Example: "The meeting date is required, attendees has 1+ people,
        key decisions have a few sentences of context, action items need
        an owner and due date — does that work?"

        After fields, derive validation rules from quality criteria.
        Example: "Since minutes should be catch-up-in-two-minutes, should
        each decision include enough context to stand on its own?"

        Use "continue" to propose, "success" when confirmed. Never mix.
        Frame in business terms. Never mention validators or Pydantic.
        """
        ...  # type: ignore[reportReturnType]
