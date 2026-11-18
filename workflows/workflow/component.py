from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from chat_workflow import composite_workflow_step
from chat_workflow.code_generator import verify_code
from chat_workflow.mixins import LLMValidated
from chat_workflow.session import Session

from . import ComponentSourceCode
from .component_spec import ComponentSpec
from .design_spec import ComponentDesignSpec
from .domain_spec import ComponentDomainSpec
from .interaction_context import ComponentInteractionContext
from .structure import ComponentStructure


class Component(LLMValidated):
    """A generated component written to disk.

    Represents a business artifact and the set of workflow steps that create
    or update it. The Component's create() method calls
    ComponentSourceCode.generate() to produce the code, then verifies, writes,
    and validates the result.
    """

    _validation_rules: ClassVar[list[str]] = [
        "The purpose field clearly describes the domain concept the component "
        "represents, in a way that an LLM could use to understand what instances "
        "of this component mean (not just what they store).",
        "The name is a noun — it names a concrete artifact (e.g. 'Invoice', "
        "'MeetingMinutes'), not an action (e.g. 'GenerateInvoice') or an "
        "abstract activity ('Approval', 'Distribution', 'Review').",
        "The expert_role is the agent's domain expertise — the role the agent "
        "acts as when helping with this artifact. It may equal or differ from "
        "the owner_role; it is not a generic role ('MinutesDraft Expert' is "
        "vague; 'Meeting Minutes Administrator' is better).",
        "The owner_role is the capacity the artifact belongs to — the role "
        "whose job includes it, which may be the user. 'Owner' only when the "
        "artifact is the owner's own work; not the person who supplies the "
        "input unless they are accountable.",
        # Tier 1 — Core Structural Integrity
        "Single Artifact Type Rule: The component owns exactly one artifact — "
        "the set of data members of the model it defines. The purpose must "
        "describe that one artifact concept, and its methods may create, "
        "update, review, or send it: those are operations on the same "
        "artifact, not additional artifacts. "
        "BAD: 'Invoice processing pipeline with integrated timesheet management' "
        "(two artifact types: invoices and timesheets). "
        "GOOD: 'Processes customer invoices through their complete lifecycle'.",
        "Single Responsibility: The purpose states this component's own "
        "responsibility, not a bundle of subcomponents, existing or latent. "
        "It describes the one artifact the component owns and the operations "
        "it performs on it. Work it delegates to other components that own "
        "their own artifacts is allowed in the purpose, but must not be "
        "restated as its own work. A piece of work that is itself an artifact — a "
        "section with its own content, audience, or quality criteria (a "
        "strategy section, a financial projection), a separate record or "
        "deliverable, or unrelated technical work such as parsing or file "
        "handling — is a latent subcomponent and does not belong here. "
        "BAD: 'Business plan: market analysis, financials, operations, and "
        "file parsing' (four subcomponents in one component). "
        "GOOD: 'The business plan document: a coherent plan assembled from "
        "components that each own one section'.",
        "No Multiple Artifact Creation: The component owns exactly one primary "
        "artifact type. Creating and updating that artifact, or recording "
        "several of its operations, is still one artifact. The purpose must "
        "not imply ownership of multiple distinct business artifacts. "
        "BAD: 'Creates customer invoices generates monthly reports produces "
        "analytics dashboards' (three distinct artifacts). "
        "GOOD: 'Creates and updates customer invoices from start to finish'.",
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
        min_length=1,
    )

    purpose: str = Field(
        ...,
        description=(
            "This component's own atomic responsibility — what the component "
            "itself does, with no subcomponents, existing or latent"
        ),
        min_length=1,
    )

    owner_role: str = Field(
        ...,
        description=(
            "The capacity the artifact belongs to — the role whose job "
            "includes it. It may be the user. 'Owner' only when the artifact "
            "is the owner's own work."
        ),
        min_length=1,
    )

    code_path: Path = Field(
        ...,
        description="Path to the generated Python file on disk",
    )

    model_class: str = Field(
        ...,
        description="Name of the Pydantic model class defined in the generated file",
        min_length=1,
    )

    expert_role: str = Field(
        ...,
        description=(
            "The agent's domain expertise — the role the agent acts as when "
            "helping with this component; may equal owner_role"
        ),
        min_length=1,
    )

    component_type: Literal["value_stream", "artifact_producing", "planning_service"] = Field(
        ...,
        description=(
            "One of: value_stream (continuous outcome or KPI, not a discrete "
            "document), artifact_producing, planning_service"
        ),
    )

    execution_complexity: Literal["simple", "complex"] = Field(
        default="simple",
        description="simple or complex — indicates implementation effort",
    )

    @model_validator(mode="after")
    def _check_single_responsibility(self) -> Component:
        """Enforce single responsibility: purpose must not contain 'also'.

        This is a fast programmatic heuristic that catches obvious violations.
        Deeper semantic checks are handled by _validation_rules (LLM-judged).

        Note: We do NOT check for 'and' here because it's commonly used in
        natural language descriptions of a single responsibility (e.g.,
        'including discussions, decisions, and action items'). The LLM-judged
        _validation_rules handle the semantic check for multiple responsibilities.
        """
        purpose_lower = self.purpose.lower()
        if re.search(r"\balso\b", purpose_lower):
            raise ValueError(
                f"Purpose must describe a single responsibility — remove conjunctions like 'also': '{self.purpose}'"
            )
        return self

    @composite_workflow_step
    @classmethod
    def create(
        cls,
        requirements: ComponentSpec | ComponentSpec,
        *,
        session: Session,
        output_dir: Path | None = None,
        deliverables: list | None = None,
    ) -> Component:
        """Create ONE business component.

        Two code paths:
        - ComponentSpec (new): Phase 1 (DomainSpec.explore)
          + Phase 2 (Structure.design) — produces a domain spec, designs
          the structure, then builds and returns a Component record.
        - ComponentSpec (legacy): Full pipeline — generates code,
          verifies, writes to disk, and returns Component.

        Args:
            requirements: The component requirements or responsibilities
            session: The chat-workflow session
            output_dir: Directory to write the component file. Defaults to
                current working directory / "workflows" / {name} /
        """
        # --- New path: ComponentSpec (Phase 1-4) ---
        if isinstance(requirements, ComponentSpec):
            session.io.echo(f"Exploring domain: {requirements.name}...")
            domain_spec = ComponentDomainSpec.explore(
                responsibilities=requirements,
                deliverables=deliverables,
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
            generated = ComponentSourceCode.generate(
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
                owner_role=requirements.owner_role,
                expert_role=domain_spec.expert_role,
                component_type=requirements.component_type,
            )
            return result

        # --- Legacy path: ComponentSpec ---
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
        generated = ComponentSourceCode.generate(
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
            owner_role=requirements.owner_role,
            expert_role=f"{requirements.name} Expert",
            component_type=requirements.component_type,
        )
        return result
