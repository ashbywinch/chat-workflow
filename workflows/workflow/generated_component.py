from __future__ import annotations

import ast
from typing import Annotated, ClassVar

from pydantic import Field, model_validator

from chat_workflow import atomic_workflow
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import LLMValidated

from .design_spec import ComponentDesignSpec

_CONSTRAINT_NAMES = frozenset({"min_length", "ge", "le", "gt", "lt", "pattern", "max_length", "multiple_of"})


def _has_field_constraint(node: ast.AST) -> bool:
    """Check if an AST node is a Field(...) call with constraint keyword arguments."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "Field":
        for kw in node.keywords:
            if kw.arg in _CONSTRAINT_NAMES:
                return True
    return False


def _is_model_validator(node: ast.AST) -> bool:
    """Check if an AST node is a function decorated with @model_validator."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "model_validator":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "model_validator":
            return True
    return False


class GeneratedComponent(LLMValidated):
    """Python source code for a chat-workflow business component.

    Represents a generated Pydantic BaseModel class that owns a single
    business artifact type and provides methods to create instances of it
    through conversational workflows. The generated class carries an
    @atomic_workflow method whose docstring becomes the system prompt for
    the LLM conversation that creates those instances.
    """

    _validation_rules: ClassVar[list[str]] = [
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
        "REJECT if the generated BaseModel has fields from multiple unrelated "
        "business domains (e.g. invoice_number AND employee_salary AND "
        "meeting_room in the same class). All fields must serve one artifact.",
        "REJECT if the class docstring or @atomic_workflow docstring is "
        "generic boilerplate longer than 3 sentences that doesn't describe "
        "the specific business artifact (e.g. 'represents a business concept' "
        "or 'captures important information' without domain specifics).",
    ]

    @model_validator(mode="after")
    def _validate_code_structure(self) -> GeneratedComponent:
        """Programmatic validation of generated code structure.

        Checks that the generated code is syntactically valid Python,
        defines exactly one BaseModel subclass, has an
        @atomic_workflow-decorated method, includes business validation,
        and has a single @atomic_workflow method. These checks run before
        the LLM-judged _validation_rules to fail fast on structural issues.
        """
        code = self.code

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValidationError(f"Generated code is not valid Python: {e}") from e

        base_model_subclasses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases)
        ]
        if len(base_model_subclasses) == 0:
            raise ValidationError("Generated code must define at least one BaseModel subclass")
        if len(base_model_subclasses) > 1:
            raise ValidationError(
                "Generated code defines multiple BaseModel subclasses "
                f"({[c.name for c in base_model_subclasses]}). "
                "A component must define exactly one cohesive business "
                "artifact type — split multiple types into separate components."
            )

        workflow_methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(d, ast.Name) and d.id == "atomic_workflow" for d in node.decorator_list)
        ]
        if len(workflow_methods) == 0:
            raise ValidationError(
                "Generated code must have a method decorated with @atomic_workflow (imported from chat_workflow)"
            )
        if len(workflow_methods) > 1:
            raise ValidationError(
                f"Generated code has {len(workflow_methods)} @atomic_workflow methods "
                f"({[m.name for m in workflow_methods]}). "
                "A component must have exactly one @atomic_workflow method — "
                "multiple workflow methods for different artifacts should be "
                "separate components."
            )

        has_business_validation = any(
            _has_field_constraint(node) or _is_model_validator(node) for node in ast.walk(tree)
        )
        if not has_business_validation:
            raise ValidationError(
                "Generated code must include business validation — either Field "
                "constraints (min_length, ge, le, pattern) or @model_validator "
                "methods that encode domain-specific business rules."
            )

        return self

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
