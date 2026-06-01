from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from chat_workflow import composite_workflow
from chat_workflow.code_generator import verify_code
from chat_workflow.mixins import LLMValidated
from chat_workflow.session import Session

from . import GeneratedComponent
from .component_responsibilities import ComponentResponsibilities
from .design_spec import ComponentDesignSpec
from .domain_spec import ComponentDomainSpec
from .interaction_context import ComponentInteractionContext
from .models import ComponentRequirement
from .structure import ComponentStructure


class Component(LLMValidated):
    """Represents a business component written to disk.

    Wraps a generated Pydantic BaseModel class that owns a single
    business artifact type. The Component's create() method calls
    GeneratedComponent.generate() to produce the code, then verifies,
    writes, and validates the result.
    """

    _validation_rules: ClassVar[list[str]] = [
        "The purpose field clearly describes the domain concept the component "
        "represents, in a way that an LLM could use to understand what instances "
        "of this component mean (not just what they store).",
        "The name is a noun — it names an artifact (e.g. 'Invoice', "
        "'MeetingMinutes'), not an action (e.g. 'GenerateInvoice').",
        "The expert_role describes a real, specific domain expertise the component "
        "embodies, not a generic role ('MinutesDraft Expert' is vague; "
        "'Meeting Minutes Administrator' is better).",
        # Tier 1 — Core Structural Integrity
        "Single Artifact Type Rule: The component defines exactly one business "
        "artifact type. The purpose must describe a single artifact concept. "
        "BAD: 'Invoice processing pipeline with integrated timesheet management' "
        "(two artifact types: invoices and timesheets). "
        "GOOD: 'Processes customer invoices through their complete lifecycle'.",
        "Single Responsibility: The component's responsibility must be stateable "
        "in one sentence describing exactly one domain concept. The purpose must "
        "not describe multiple distinct responsibilities. "
        "BAD: 'Oversees customer onboarding handles billing inquiries manages "
        "support tickets' (three distinct responsibilities). "
        "GOOD: 'Processes customer invoices through their complete lifecycle'.",
        "No Multiple Artifact Creation: The component creates exactly one primary "
        "artifact type. The purpose must not imply creation of multiple distinct "
        "business artifacts. "
        "BAD: 'Creates customer invoices generates monthly reports produces "
        "analytics dashboards' (three distinct artifacts). "
        "GOOD: 'Creates customer invoices from start to finish'.",
        "Clear Boundaries: The purpose must clearly define what is inside and "
        "outside the component's responsibility. Another LLM should be able to "
        "use the purpose to decide if a given concern belongs here. "
        "BAD: 'Handles everything related to the business operations of the "
        "company' (no clear boundary). "
        "GOOD: 'Creates customer invoices from submission through final "
        "distribution' (clear start and end).",
        "Encapsulation: The component's fields and methods must all relate to "
        "the same domain concept. The purpose must not mix unrelated concerns. "
        "BAD: 'Manages user authentication database backups email notifications' "
        "(three unrelated domains). "
        "GOOD: 'Creates customer invoices from submission through final "
        "distribution' (single domain).",
        "Cohesion: All functionality described in the purpose must serve the "
        "same primary artifact. The purpose must not describe orphaned "
        "functionality that belongs to a different domain. "
        "BAD: 'Handles email notifications performs database maintenance' "
        "(functionality serves no common artifact). "
        "GOOD: 'Creates customer invoices from submission through final "
        "distribution' (all serves invoice artifact).",
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

    @model_validator(mode="after")
    def _check_single_responsibility(self) -> Component:
        """Enforce single responsibility: purpose must not contain 'and' or 'also'.

        This is a fast programmatic heuristic that catches obvious violations.
        Deeper semantic checks are handled by _validation_rules (LLM-judged).
        """
        purpose_lower = self.purpose.lower()
        if re.search(r"\band\b", purpose_lower):
            raise ValueError(
                f"Purpose must describe a single responsibility — remove conjunctions like 'and': '{self.purpose}'"
            )
        if re.search(r"\balso\b", purpose_lower):
            raise ValueError(
                f"Purpose must describe a single responsibility — remove conjunctions like 'also': '{self.purpose}'"
            )
        return self

    @composite_workflow
    @classmethod
    def create(
        cls,
        requirements: ComponentRequirement | ComponentResponsibilities,
        *,
        session: Session,
        output_dir: Path | None = None,
    ) -> Component:
        """Create ONE business component.

        Two code paths:
        - ComponentResponsibilities (new): Phase 1 (DomainSpec.explore)
          + Phase 2 (Structure.design) — produces a domain spec, designs
          the structure, then builds and returns a Component record.
        - ComponentRequirement (legacy): Full pipeline — generates code,
          verifies, writes to disk, and returns Component.

        Args:
            requirements: The component requirements or responsibilities
            session: The chat-workflow session
            output_dir: Directory to write the component file. Defaults to
                current working directory / "workflows" / {name} /
        """
        # --- New path: ComponentResponsibilities (Phase 1-4) ---
        if isinstance(requirements, ComponentResponsibilities):
            session.io.echo(f"Exploring domain: {requirements.name}...")
            domain_spec = ComponentDomainSpec.explore(
                responsibilities=requirements,
                session=session,
            )

            session.io.echo(f"Designing structure: {domain_spec.name}...")
            structure = ComponentStructure.design(
                domain_spec=domain_spec,
                session=session,
            )

            session.io.echo(f"Gathering interaction context: {domain_spec.name}...")
            interaction_context = ComponentInteractionContext.gather(
                domain_spec=domain_spec,
                structure=structure,
                session=session,
            )

            design_spec = ComponentDesignSpec(
                domain_spec=domain_spec,
                structure=structure,
                interaction_context=interaction_context,
            )

            session.io.echo(f"Generating component code: {domain_spec.name}...")
            generated = GeneratedComponent.generate(
                design_spec=design_spec,
                session=session,
            )

            session.io.echo("Verifying generated code...")
            try:
                clean_code = verify_code(generated.code)
            except RuntimeError as e:
                session.io.echo(f"Code verification failed: {e}")
                raise

            if output_dir is None:
                output_dir = Path.cwd() / "workflows" / domain_spec.name.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            code_path = output_dir / f"{domain_spec.name.lower()}.py"

            code_path.write_text(clean_code)
            session.io.echo(f"Component written to: {code_path}")

            result = cls(
                name=domain_spec.name,
                purpose=structure.description,
                code_path=code_path,
                model_class=domain_spec.name,
                expert_role=domain_spec.expert_role,
                component_type=requirements.component_type,
            )
            return result

        # --- Legacy path: ComponentRequirement ---
        # Step 1: Construct a ComponentDesignSpec from the requirement
        session.io.echo(f"Designing component: {requirements.name}...")
        design_spec = ComponentDesignSpec(
            domain_spec=ComponentDomainSpec(
                name=requirements.name,
                description=requirements.purpose,
                fields=[],
                what_good_looks_like=[],
                expert_role=f"{requirements.name} Expert",
            ),
            structure=ComponentStructure(
                description=requirements.purpose,
            ),
            interaction_context=ComponentInteractionContext(
                must_prioritize=[],
                auto_suggest=[],
                user_pain_points=[],
            ),
        )
        generated = GeneratedComponent.generate(
            design_spec=design_spec,
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

        result = cls(
            name=requirements.name,
            purpose=requirements.purpose,
            code_path=code_path,
            model_class=requirements.name,
            expert_role=f"{requirements.name} Expert",
            component_type=requirements.component_type,
        )
        return result
