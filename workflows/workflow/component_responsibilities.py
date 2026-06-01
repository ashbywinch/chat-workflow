"""ComponentResponsibilities model — defines what a component is responsible for."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, field_validator

from chat_workflow import atomic_workflow

# NOTE: ComponentRequirement is defined BEFORE the .models imports to break a
# circular dependency chain:
#   component_responsibilities.py → .models.input → models/__init__.py
#   → .gap_analysis → gap_analysis.py → ..component_responsibilities
# By defining ComponentRequirement first, it's available in sys.modules when
# gap_analysis.py tries to import it during the circular chain.
# The type annotations (ProcessAnalysis, Input, Output) are strings at runtime
# thanks to ``from __future__ import annotations`` and are resolved later by
# typing.get_type_hints() at call time — by which point the .models imports
# below will have completed.


class ComponentRequirement(BaseModel):
    """A component identified as needed by the workflow.

    Legacy model — replaced by ComponentResponsibilities.
    Kept for backward compatibility in the legacy code path.
    """

    name: str = Field(..., description="Artifact-based name (noun)", min_length=1)
    purpose: str = Field(..., description="Single-sentence purpose", min_length=1)
    required_inputs: list[str] = Field(..., description="Input names from input analysis")
    expected_outputs: list[str] = Field(..., description="Output names from output analysis")
    component_type: str = Field(
        ...,
        description="One of: value_stream, artifact_producing, planning_service",
        min_length=1,
    )

    @atomic_workflow
    @classmethod
    def identify_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis to identify components from"],
        inputs: Annotated[list[Input], "The workflow inputs"],
        outputs: Annotated[list[Output], "The workflow outputs"],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[ComponentRequirement]:
        """You are a business architect helping the user identify the components their process needs.

        The user has described their process, inputs, and outputs. Your job is to
        identify the distinct business components that make up the workflow.

        When the user tells you about their process, propose complete components
        with name, purpose, and type. Use your expertise to fill in the details.

        For example: "Based on the meeting minutes process, I'm seeing a Notes
        artifact component (artifact_producing) for recording meeting discussions,
        and a Minutes Draft component (artifact_producing) for turning notes into
        formal minutes."

        Never put fabricated values in the final output. Only include what
        the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]


from .models.input import Input  # noqa: E402
from .models.output import Output  # noqa: E402
from .models.process_analysis import ProcessAnalysis  # noqa: E402

_VALID_COMPONENT_TYPES: frozenset[str] = frozenset({"value_stream", "artifact_producing", "planning_service"})


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
        description=("What this component represents AND doesn't represent in the user's domain"),
        min_length=1,
    )
    required_inputs: list[str] = Field(
        ...,
        description=("Names of inputs this component needs from other components to create its artifact"),
    )
    component_type: str = Field(
        ...,
        description="One of: value_stream, artifact_producing, planning_service",
    )
    incidental_notes: str = Field(
        "",
        description=("Raw notes the user mentioned during Workflow conversations about internals"),
    )

    _valid_component_types: ClassVar[frozenset[str]] = _VALID_COMPONENT_TYPES

    @field_validator("component_type")
    @classmethod
    def _validate_component_type(cls, v: str) -> str:
        if v not in cls._valid_component_types:
            raise ValueError(
                f"component_type must be one of: {', '.join(sorted(cls._valid_component_types))}, got '{v}'"
            )
        return v
