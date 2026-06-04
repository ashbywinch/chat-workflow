from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow
from chat_workflow.mixins import LLMValidated

from ..component_responsibilities import ComponentResponsibilities


class ComponentSourceCode(LLMValidated):
    """Python source code for a chat-workflow business component.

    Represents a generated Pydantic BaseModel class that owns a single
    business artifact type and provides methods to create instances of it
    through conversational workflows. The generated class carries an
    @atomic_workflow method whose docstring becomes the system prompt for
    the LLM conversation that creates those instances.
    """

    _validation_rules: ClassVar[list[str]] = [
        "The code is valid Python that defines at least one BaseModel subclass.",
        "The generated BaseModel uses Field(..., description=...) on each field "
        "with a plain-English description of what the field means in the "
        "business domain.",
        "Field definitions on the generated class use min_length=1 on string "
        "or collection fields where an empty value would be semantically "
        "meaningless for that business concept.",
        "Validation in the generated code (Field constraints or "
        "@model_validator) encodes business rules a domain expert would care "
        "about — not just type checks.",
        "The generated class has an @atomic_workflow classmethod whose "
        "docstring tells the agent to propose and synthesize rather than "
        "asking the user to fill out a form, and includes an example of "
        "the desired conversational rhythm.",
    ]

    code: str = Field(
        ...,
        description="Complete Python source code defining the component: "
        "a Pydantic BaseModel class with an @atomic_workflow classmethod "
        "whose docstring is a good system prompt (proposes and synthesizes, "
        "includes a dialogue example), field definitions with clear business "
        "descriptions, and validation rules encoding real business semantics.",
        min_length=1,
    )

    @atomic_workflow
    @classmethod
    def generate(
        cls,
        requirements: Annotated[
            ComponentResponsibilities,
            "The component requirements specifying name, purpose, inputs, outputs, and type",
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ComponentSourceCode:
        """You are a software architect designing a Python business component.

        The user has described what they need. Your job is to generate Python code
        that defines a Pydantic BaseModel class — a business component — with:
        - A class docstring that explains the domain concept the model represents,
          in terms an LLM with no context could understand
        - An @atomic_workflow classmethod whose docstring is a good system prompt:
          it frames the task around the user's domain, tells the agent to propose
          and synthesize, and includes a concrete example of the conversational
          rhythm (e.g., "From what you've described, I'm seeing...")
        - Field definitions using Field(..., description=...) with plain-English
          business descriptions and min_length=1 where empty values are meaningless
        - Validation rules (Field constraints or @model_validator) that encode
          real business semantics, not just type checks

        Design the component conversationally:
        - Propose concrete fields, data types, and validation rules based on what
          the user has told you. Use your expertise to fill in the gaps.
        - When the user describes a rule or constraint, summarize and build on it.
          For example: "So every action item needs an owner and due date — I'll make
          those required fields with validation. What about the description length?"
        - Never put fabricated validation rules in the final output. Only include
          what the user has confirmed. But you can propose ideas in conversation.

        Code generation rules:
        - Import from __future__ import annotations, pydantic BaseModel and Field
        - One class per file named after the component
        - Valid Python that passes ruff linting
        - Keep all lines under 120 characters — break long docstrings and
          Field descriptions into multiple short lines

        Output format: Return ONLY the Python code as a string in the 'code' field.
        """
        ...  # type: ignore[reportReturnType]
