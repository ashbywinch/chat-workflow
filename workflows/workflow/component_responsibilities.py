"""ComponentResponsibilities model — defines what a component is responsible for."""

from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

_VALID_COMPONENT_TYPES: frozenset[str] = frozenset(
    {"value_stream", "artifact_producing", "planning_service"}
)


class ComponentResponsibilities(BaseModel):
    """What a component is responsible for in the user's domain.

    This replaces ComponentRequirement. The component's own model IS its
    output — there is no separate ``expected_outputs`` field.
    """

    name: str = Field(
        ...,
        description="Artifact-based name (noun), e.g. 'Invoice' not 'GenerateInvoice'",
        min_length=1,
    )
    purpose: str = Field(
        ...,
        description="Domain purpose in one sentence",
        min_length=1,
    )
    scope_description: str = Field(
        ...,
        description=(
            "What this component represents AND doesn't represent "
            "in the user's domain"
        ),
        min_length=1,
    )
    required_inputs: list[str] = Field(
        ...,
        description=(
            "Names of inputs this component needs from other components "
            "to create its artifact"
        ),
    )
    component_type: str = Field(
        ...,
        description="One of: value_stream, artifact_producing, planning_service",
    )
    incidental_notes: str = Field(
        "",
        description=(
            "Raw notes the user mentioned during Workflow conversations "
            "about internals"
        ),
    )

    _valid_component_types: ClassVar[frozenset[str]] = _VALID_COMPONENT_TYPES

    @field_validator("component_type")
    @classmethod
    def _validate_component_type(cls, v: str) -> str:
        if v not in cls._valid_component_types:
            raise ValueError(
                f"component_type must be one of: "
                f"{', '.join(sorted(cls._valid_component_types))}, got '{v}'"
            )
        return v
