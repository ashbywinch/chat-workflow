from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow
from chat_workflow.mixins import LLMValidated

from .design_spec import ComponentDesignSpec


class GeneratedComponent(LLMValidated):
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
        design_spec: Annotated[
            ComponentDesignSpec,
            "Complete component design specification — domain understanding, "
            "Pydantic structure, and interaction context. The design is fully "
            "assembled; no further user input is needed.",
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 3,
    ) -> GeneratedComponent:
        """You are a Python code generator. You have been given a complete
        component design specification. Your job is to generate valid Python
        source code that implements this design.

        The design spec contains everything you need:
        - Domain specification: what the artifact represents in the user's world,
          its fields (names, descriptions, type hints), and holistic quality criteria
        - Structure: the Pydantic model fields with Python types, Field() kwargs,
          validation rules, and any extra imports needed
        - Interaction context: how the assistant should interact with users when
          creating instances of this artifact

        Generate a complete Python module that defines:
        1. A Pydantic BaseModel class (or LLMValidated subclass) named after the
           component, with a class docstring explaining the domain concept
        2. A classmethod decorated with @atomic_workflow (imported from
           chat_workflow) whose docstring is a good system prompt — it frames the
           task around the user's domain, tells the agent to propose and synthesize,
           and includes a concrete dialogue example.
        3. Field definitions using Field(..., description=...) with plain-English
           business descriptions and appropriate constraints (min_length, etc.)
        4. Validation rules (@model_validator or Field constraints) that encode
           real business semantics from the quality criteria

        Code generation rules:
        - Import atomic_workflow from chat_workflow (just the name, no alias)
        - Use @atomic_workflow on its own line WITHOUT parentheses or arguments
          (correct: "@atomic_workflow" then "@classmethod" then "def method_name")
        - Import from __future__ import annotations, pydantic BaseModel and Field
        - One class per file named after the component
        - Valid Python that passes ruff linting

        The design is complete. Do NOT ask questions, propose alternatives, or
        request confirmation. Just generate the code and return it immediately
        with intent "success".

        Output format: Return ONLY the Python code as a string in the 'code' field.
        """
        ...  # type: ignore[reportReturnType]