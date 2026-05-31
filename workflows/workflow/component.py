from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow, composite_workflow
from chat_workflow.code_generator import verify_code
from chat_workflow.mixins import LLMValidated
from chat_workflow.session import Session

from .models import ComponentRequirement, GeneratedComponent


class Component(LLMValidated):
    """A created business component.

    Represents a single business component that has been designed and
    written to disk by the component workflow.
    """

    _validation_rules: ClassVar[list[str]] = [
        "Generated class must inherit from BaseModel",
        "Generated class must have a generate_from_chat classmethod",
        "All field types must be valid Python types",
    ]

    name: str = Field(
        ...,
        description="Component name (primary artifact, noun-based)",
    )

    purpose: str = Field(
        ...,
        description="Component purpose and capabilities in one sentence",
    )

    code_path: Path = Field(
        ...,
        description="Path to the generated Python file on disk",
    )

    model_class: str = Field(
        ...,
        description="Name of the Pydantic model class defined in the generated file",
    )

    expert_role: str = Field(
        ...,
        description="Single expert role responsible for this component",
    )

    component_type: str = Field(
        ...,
        description="One of: value_stream, artifact_producing, planning_service",
    )

    execution_complexity: str = Field(
        default="simple",
        description="simple or complex — indicates implementation effort",
    )

    @composite_workflow
    @classmethod
    def create(
        cls,
        requirements: ComponentRequirement,
        *,
        session: Session,
        output_dir: Path | None = None,
    ) -> Component:
        """Create ONE business component.
        
        1. _design_component(requirements) -> GeneratedComponent with code
        2. verify_code(code) -> clean, formatted code
        3. Write to file at output_dir / {name}.py
        4. Build, validate, and return Component object with code_path set
        
        Args:
            requirements: The component requirements
            session: The chat-workflow session
            output_dir: Directory to write the component file. Defaults to
                current working directory / "workflows" / {name} /
        """
        # Step 1: Design the component (LLM generates code)
        session.io.echo(f"Designing component: {requirements.name}...")
        generated = cls._design_component(
            requirements=requirements,
            session=session,
        )

        # Step 2: Verify code quality
        session.io.echo("Verifying generated code...")
        try:
            clean_code = verify_code(generated.code)
        except RuntimeError as e:
            session.io.echo(f"Code verification failed: {e}")
            raise

        # Step 3: Determine output path
        if output_dir is None:
            output_dir = Path.cwd() / "workflows" / requirements.name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        code_path = output_dir / f"{requirements.name.lower()}.py"

        # Step 4: Write the file
        code_path.write_text(clean_code)
        session.io.echo(f"Component written to: {code_path}")

        # Step 5: Validate the Component object (triggers LLMValidated @model_validator)
        try:
            result = cls(
                name=requirements.name,
                purpose=requirements.purpose,
                code_path=code_path,
                model_class=requirements.name,
                expert_role=f"{requirements.name} Expert",
                component_type=requirements.component_type,
            )
            result.validate_llm_rules()
            return result
        except Exception as e:
            session.io.echo(f"Validation failed: {e}")
            raise

    @atomic_workflow
    @classmethod
    def _design_component(
        cls,
        requirements: Annotated[
            ComponentRequirement,
            "The component requirements specifying name, purpose, inputs, outputs, and type",
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> GeneratedComponent:
        """You are a software architect designing a Python business component with a user.

        The user has described what they need. Your job is to design the component —
        the fields it needs, the validation rules its data should follow — and generate
        the Python code.

        This isn't a form to fill out. You're an expert software architect. Listen to
        what the user tells you about their domain, propose what you think the fields
        and rules should be, and check your understanding with them.

        - When the user describes a rule or constraint, summarize and build on it.
          For example: "So every action item needs an owner and due date — I'll make
          those required fields with validation. What about the description length?"
        - Propose concrete fields, data types, and validation rules based on what
          the user has told you. Use your expertise to fill in the gaps.
        - Never put fabricated validation rules in the final output. Only include
          what the user has confirmed. But you can propose ideas in conversation.
        - Use your expertise to generate clean Python code with proper imports,
          BaseModel inheritance, Field definitions with descriptions, and validation.

        Code rules:
        - Import from __future__ import annotations, pydantic BaseModel and Field
        - One class per file named after the component
        - Valid Python that passes ruff linting

        Output format: Return ONLY the Python code as a string in the 'code' field.
        """
        ...  # type: ignore[reportReturnType]