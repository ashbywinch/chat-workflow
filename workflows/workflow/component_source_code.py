from __future__ import annotations

import ast
from typing import Annotated, ClassVar

from pydantic import Field, model_validator

from chat_workflow import atomic_workflow
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import LLMValidated

from .design_spec import ComponentDesignSpec

_CONSTRAINT_NAMES = frozenset({
    "min_length", "max_length",
    "min_items", "max_items",
    "ge", "le", "gt", "lt",
    "pattern", "multiple_of",
})


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
        # Bare @model_validator
        if isinstance(decorator, ast.Name) and decorator.id == "model_validator":
            return True
        # @model_validator(mode="after") — call form
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "model_validator":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "model_validator":
                return True
        # Qualified name like chat_workflow.model_validator
        if isinstance(decorator, ast.Attribute) and decorator.attr == "model_validator":
            return True
    return False


class ComponentSourceCode(LLMValidated):
    """Python source code for a chat-workflow business component.

    Represents a generated Pydantic BaseModel class that owns a single
    business artifact type and provides methods to create instances of it
    through conversational workflows. The generated class carries an
    @atomic_workflow method whose docstring becomes the system prompt for
    the LLM conversation that creates those instances.
    """

    _validation_rules: ClassVar[list[str]] = [
        "Every field must use Field(..., description=...) with a "
        "plain-English description of what the field means in the "
        "business domain.",
        "String or collection fields where an empty value would make "
        "the model categorically invalid must have min_length or "
        "min_items constraints. An empty 'name' string is nonsense; "
        "an empty list of gaps simply means 'no gaps found.'",
        "Validation in the source code (Field constraints or "
        "@model_validator) encodes business rules a domain expert would "
        "care about — not just type checks.",
        "The workflow docstring describes what a great conversation looks "
        "like in this context — how the agent should engage, probe, and "
        "guide the user to produce a valid return value from the parameters.",
        "REJECT if the workflow docstring duplicates information from the "
        "class docstring, field definitions, or validation rules.",
        "Every word in every prompt string must earn its place.",
        "Validation logic that can be expressed as Pydantic field types, "
        "constraints (min_length, ge, pattern), or object structure must "
        "use those structural mechanisms — not LLM-judged rules (e.g. "
        "use a field with min_length=1 rather than an LLM rule saying "
        "'name must not be empty'). LLM-judged rules should only be used "
        "for quality criteria that require domain judgment: specificity, "
        "domain fit, and excellence by user standards.",
        "REJECT if the Pydantic model has fields from multiple unrelated "
        "business domains (e.g. invoice_number AND employee_salary AND "
        "meeting_room in the same class). All fields must serve one artifact.",
        "REJECT if the class docstring describes the model's fields, "
        "structure, or validation rules.",
        "Each _validation_rules entry must express exactly one rule, stated "
        "positively ('Do X') or negatively ('REJECT if Y').",
    ]

    @model_validator(mode="after")
    def _validate_code_structure(self) -> ComponentSourceCode:
        """Programmatic validation before LLM-judged rules.

        Syntax check, runtime BaseModel subclass check (resolves indirect
        inheritance like LLMValidated → BaseModel), public workflow method,
        return type check, and business validation presence check.
        """
        code = self.code

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValidationError(f"Generated code is not valid Python: {e}") from e

        # Find BaseModel subclasses — try runtime issubclass first, fall back to AST
        try:
            namespace: dict = {}
            exec(compile(tree, "<generated>", "exec"), namespace)
            from pydantic import BaseModel

            base_model_subclasses = [
                obj
                for obj in namespace.values()
                if isinstance(obj, type) and obj is not BaseModel and issubclass(obj, BaseModel)
            ]
            runtime_available = True
        except Exception:
            # AST fallback: resolve which imports bring BaseModel subtypes into scope
            _base_model_sources = frozenset({"pydantic", "chat_workflow.mixins"})
            known_basemodel_names: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in _base_model_sources:
                    for alias in node.names:
                        known_basemodel_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in _base_model_sources:
                            known_basemodel_names.add(alias.asname or alias.name.rsplit(".", 1)[-1])

            base_model_subclasses = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id in known_basemodel_names
                    for base in node.bases
                )
            ]
            runtime_available = False

        if len(base_model_subclasses) == 0:
            raise ValidationError("Generated code must define at least one BaseModel subclass")
        if len(base_model_subclasses) > 1:
            names = [
                c.__name__ if isinstance(c, type) else c.name
                for c in base_model_subclasses
            ]
            raise ValidationError(
                "Generated code defines multiple BaseModel subclasses "
                f"({names}). A component must define exactly one cohesive "
                "business artifact type — split multiple types into "
                "separate components."
            )

        main_cls = base_model_subclasses[0]

        # Find public workflow methods (@atomic_workflow / @composite_workflow)
        workflow_methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            and any(
                (
                    isinstance(d, ast.Name)
                    and d.id in ("atomic_workflow", "composite_workflow")
                )
                or (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id in ("atomic_workflow", "composite_workflow")
                )
                for d in node.decorator_list
            )
        ]

        if len(workflow_methods) == 0:
            raise ValidationError(
                "Generated code must have at least one public method decorated with "
                "@atomic_workflow or @composite_workflow (imported from chat_workflow)"
            )

        # Check that at least one method returns the class type
        main_cls_name = main_cls.__name__ if isinstance(main_cls, type) else main_cls.name
        returns_class_type = any(
            (
                isinstance(n.returns, ast.Name) and n.returns.id == main_cls_name
            )
            or (
                isinstance(n.returns, ast.Constant)
                and isinstance(n.returns.value, str)
                and n.returns.value == main_cls_name
            )
            or (
                isinstance(n.returns, ast.Subscript)
                and isinstance(n.returns.value, ast.Name)
                and n.returns.value.id == "list"
                and isinstance(n.returns.slice, ast.Name)
                and n.returns.slice.id == main_cls_name
            )
            or (
                isinstance(n.returns, ast.Subscript)
                and isinstance(n.returns.value, ast.Name)
                and n.returns.value.id == "Sequence"
                and isinstance(n.returns.slice, ast.Name)
                and n.returns.slice.id == main_cls_name
            )
            for n in workflow_methods
        )
        if not returns_class_type:
            raise ValidationError(
                f"Generated code must have a public workflow method with a return "
                f"type annotation of '{main_cls_name}' — e.g. "
                f"'def generate_from_chat(cls) -> {main_cls_name}'"
            )

        # Business validation check
        has_field_constraint = any(
            _has_field_constraint(node) for node in ast.walk(tree)
        )
        has_model_validator = any(
            _is_model_validator(node) for node in ast.walk(tree)
        )

        has_validation_rules = bool(
            any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "_validation_rules"
                    for t in (node.targets if hasattr(node, "targets") else [node.target])
                )
                for node in ast.walk(tree)
            )
        )
        has_validation_annotations = bool(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "chat_workflow.annotations"
                and any(alias.name == "Validation" for alias in node.names)
                for node in ast.walk(tree)
            )
        )
        # If runtime is available, supplement with runtime introspection
        if runtime_available and isinstance(main_cls, type):
            try:
                from chat_workflow.annotations import Validation

                has_validation_annotations = has_validation_annotations or any(
                    any(isinstance(meta, Validation) for meta in field.metadata)
                    for field in main_cls.model_fields.values()
                )
            except Exception:
                pass

        if not (
            has_field_constraint
            or has_model_validator
            or has_validation_rules
            or has_validation_annotations
        ):
            raise ValidationError(
                "Generated code must include business validation — Field "
                "constraints (min_length, ge, le, pattern), @model_validator "
                "methods, _validation_rules ClassVar, or Validation "
                "annotations that encode domain-specific business rules."
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
    ) -> ComponentSourceCode:
        """You are a Python code generator. You have been given a complete
        component design specification. Generate valid Python code that
        implements this design.

        The design spec contains everything you need:
        - Domain specification: what the artifact represents in the user's world,
          its fields (names, descriptions, type hints), and holistic quality criteria
        - Structure: the Pydantic model fields with Python types, Field() kwargs,
          validation rules, and any extra imports needed
        - Interaction context: how the assistant should interact with users when
          creating instances of this artifact

        Generate a complete Python module with:
        1. A Pydantic BaseModel class named after the component, with a class
           docstring explaining the domain concept
        2. An @atomic_workflow classmethod whose docstring is the system prompt
           for the conversation that creates instances of this model
        3. Field definitions with plain-English business descriptions and
           appropriate constraints
        4. Validation rules encoding real business semantics from the criteria

        The @atomic_workflow docstring is a system prompt for the agent.
        It should give the agent domain expertise relevant to the model
        being created. Write it to convey:

        - What the agent is helping the user accomplish, in the user's
          domain language. "You are helping someone document what they
          create" not "You are filling in a Deliverable model."
        - What domain expertise the agent should bring. For meeting
          minutes: "Meeting minutes typically capture date, attendees,
          decisions — use that knowledge to suggest what fits."
        - That the model itself defines what "done" looks like. The
          agent should return success when it has enough information
          to populate the model's required fields.

        Skip conversation scripts like "propose then confirm then success"
        — the agent can figure out the flow. Keep it token-efficient:
        every sentence should change behavior. Skip greetings — the
        component is called from a parent workflow context.

        Code structure rules:
        - Import atomic_workflow from chat_workflow
        - Import ONE_GUESS, SYNTHESIZES_HONESTLY, NO_EXECUTOR_MODE from
          chat_workflow.conversation_rules
        - Use @atomic_workflow(conversation_validation_rules=[ONE_GUESS,
          SYNTHESIZES_HONESTLY, NO_EXECUTOR_MODE]) on the classmethod
        - The @atomic_workflow method MUST have a return type annotation:
          ``-> ClassName`` (e.g., ``-> MeetingMinutes:``)
        - The @atomic_workflow method body MUST be just ``...`` (Ellipsis)
        - Define EXACTLY ONE BaseModel subclass — the primary business
          artifact. Use ``@dataclass`` or ``TypedDict`` for supporting
          data shapes (field types, sub-structures). Do NOT create
          additional BaseModel subclasses.
        - Import from __future__ import annotations, pydantic BaseModel/Field
        - One class per file named after the component
        - Valid Python that passes ruff linting

        The design is complete. Return the code with intent "success".
        """
        ...  # type: ignore[reportReturnType]
