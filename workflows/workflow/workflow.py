from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import Session, atomic_workflow, composite_workflow
from chat_workflow.annotations import Blob, Validation
from chat_workflow.mixins import BlobSyncMixin, LLMValidated

from .component_responsibilities import ComponentRequirement, ComponentResponsibilities
from .models import (
    GapAnalysis,
    Input,
    Output,
    ProcessAnalysis,
)
from .models.gap_analysis import GapAnalysis as _GapAnalysis


class Workflow(BlobSyncMixin, LLMValidated):
    """A complete workflow specification.

    Combines process analysis, input/output specs, component requirements,
    gap analysis, and a Mermaid sequence diagram into a single validated artifact.
    """

    name: str = Field(
        ...,
        description="Workflow name ending in 'Workflow'",
    )

    diagram: Annotated[
        str,
        Blob(".mmd"),
        Validation("Must use sequenceDiagram format with proper participant declarations"),
        Validation("Participants must use 'Component Path: Playbook Name' naming format"),
        Validation("Must not include classDef or styling directives"),
        Validation("Must use <br/> tags for splitting long text across lines"),
    ] = Field(
        ...,
        description="Mermaid sequence diagram as text (sequenceDiagram format)",
    )

    inputs: list[Input] = Field(
        ...,
        description="All workflow inputs with source, format, trigger, and validation analysis",
    )

    outputs: list[Output] = Field(
        ...,
        description="All workflow outputs with consumer, format, success criteria, and integration analysis",
    )

    components: list[ComponentResponsibilities] = Field(
        ...,
        description="Identified components required by this workflow",
    )

    gap_analysis: GapAnalysis | None = Field(
        None,
        description="Analysis of missing elements, integration gaps, and organizational gaps",
    )

    architectural_validation: str = Field(
        ...,
        description="Validation of ownership, alignment, separation of concerns, and dependency management",
    )

    _validation_rules: ClassVar[list[str]] = [
        "All component names must follow artifact-based naming (nouns, not processes)",
        "Every input must have a matching output consumer",
        "No activities lack proper component ownership",
    ]

    @atomic_workflow
    @classmethod
    def _generate_diagram(
        cls,
        analysis: Annotated[ProcessAnalysis, "The process analysis"],
        components: Annotated[list[ComponentResponsibilities], "The identified components"],
        inputs: Annotated[list[Input], "The workflow inputs"],
        outputs: Annotated[list[Output], "The workflow outputs"],
        gap_analysis: Annotated[GapAnalysis | None, "Optional gap analysis to incorporate"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> Workflow:
        """You are creating a complete workflow artifact with a Mermaid sequence diagram.

        Create a Mermaid sequenceDiagram with:
        - Participants in 'Component Path: Playbook Name' format
        - Split long text across lines using <br/> tags
        - No classDef or styling directives
        - Entry point, all interactions through orchestrator, decision points, return flows, clear outcomes

        Also populate all fields:
        - name: Workflow name ending in 'Workflow'
        - inputs: from the provided input analysis
        - outputs: from the provided output analysis
        - components: from the identified components
        - gap_analysis: from the provided gap analysis
        - architectural_validation: validate ownership, alignment, and dependencies

        Guidelines:
        - The diagram must tell a complete story of the business process
        - Every activity should be represented in the sequence flow
        - Participants should be meaningful business roles
        - Ask one question at a time
        """
        ...  # type: ignore[reportReturnType]

    @classmethod
    def _create_diagram(
        cls,
        analysis: ProcessAnalysis,
        components: list[ComponentResponsibilities],
        inputs: list[Input],
        outputs: list[Output],
        gap_analysis: GapAnalysis | None = None,
        *,
        session: Session,
        max_refinements: int = 3,
    ) -> Workflow:
        import tempfile
        from pathlib import Path

        from workflows.evaluation_criteria.refine import refine

        tmp_dir = tempfile.mkdtemp(prefix="workflow_diagram_")

        workflow = cls._generate_diagram(
            analysis=analysis,
            components=components,
            inputs=inputs,
            outputs=outputs,
            gap_analysis=gap_analysis,
            session=session,
        )

        workflow.materialize_blobs(Path(tmp_dir))
        diagram_path = workflow.get_blob_path("diagram")
        if diagram_path:
            session.io.echo(f"Diagram saved to: {diagram_path}")

        for _ in range(max_refinements):
            refined = refine(
                initial_object=workflow,
                max_turns=5,
                session=session,
            )

            # User satisfied when object returned unchanged
            if refined.model_dump() == workflow.model_dump():
                return refined

            workflow = refined
            workflow.materialize_blobs(Path(tmp_dir))
            if diagram_path:
                session.io.echo(f"Updated diagram saved to: {diagram_path}")

        return workflow

    @composite_workflow
    @classmethod
    def create(
        cls,
        process_description: str = "",
        *,
        session: Session,
        max_refinements: int = 3,
        existing_components: list[str] | None = None,
    ) -> Workflow:
        """Create a complete workflow artifact through conversation.

        Orchestrates: process analysis, component identification,
        gap resolution, diagram generation, user refinement, and
        component creation.
        """
        outputs = Output.generate_from_chat(session=session)

        session.io.echo("\nNow let's figure out what you have to work with.")
        inputs = Input.generate_from_chat(outputs=outputs, session=session)

        session.io.echo("\nLet me understand how it all fits together.")
        analysis = ProcessAnalysis.generate_from_chat(
            process_description=process_description,
            outputs=outputs,
            inputs=inputs,
            session=session,
        )

        components, gap_analysis = _resolve_gaps(
            analysis=analysis,
            inputs=inputs,
            outputs=outputs,
            existing_components=existing_components or [],
            session=session,
        )

        workflow = cls._create_diagram(
            analysis=analysis,
            components=components,
            inputs=inputs,
            outputs=outputs,
            gap_analysis=gap_analysis,
            session=session,
            max_refinements=max_refinements,
        )

        session.io.echo(f"\nCreating {len(components)} component(s)...")
        for req in components:
            session.io.echo(f"  Creating component: {req.name}")
            try:
                from .component import Component as ComponentModel

                component = ComponentModel.create(
                    requirements=req,
                    session=session,
                )
                session.io.echo(f"  ✓ {req.name} created at {component.code_path}")
            except Exception as e:
                session.io.echo(f"  ✗ Failed to create {req.name}: {e}")

        return workflow


def _requirement_to_responsibilities(
    req: ComponentRequirement,
) -> ComponentResponsibilities:
    """Convert a ComponentRequirement to ComponentResponsibilities.

    Uses the requirement's purpose as the initial scope_description.
    Incidental notes are captured separately via _capture_incidental_notes.
    """
    return ComponentResponsibilities(
        name=req.name,
        purpose=req.purpose,
        scope_description=req.purpose,
        required_inputs=req.required_inputs,
        component_type=req.component_type,
        incidental_notes="",
    )


def _requirements_to_responsibilities(
    requirements: list[ComponentRequirement],
) -> list[ComponentResponsibilities]:
    return [_requirement_to_responsibilities(r) for r in requirements]


def _capture_incidental_notes(
    components: list[ComponentResponsibilities],
    *,
    session: Session,
) -> list[ComponentResponsibilities]:
    """Ask the user about incidental notes for each component.

    During Workflow conversations, the user may mention internal details,
    implementation preferences, or constraints about a component. This
    function gives them an explicit opportunity to share those notes.
    """
    result: list[ComponentResponsibilities] = []
    for comp in components:
        session.io.echo(f"\nAny incidental notes about the '{comp.name}' component?")
        session.io.echo(
            "(e.g., internal details, implementation preferences, or constraints the component should follow)"
        )
        notes = session.io.prompt(f"Incidental notes for {comp.name}")
        result.append(
            ComponentResponsibilities(
                name=comp.name,
                purpose=comp.purpose,
                scope_description=comp.scope_description,
                required_inputs=comp.required_inputs,
                component_type=comp.component_type,
                incidental_notes=notes,
            )
        )
    return result


def _resolve_gaps(
    analysis: ProcessAnalysis,
    inputs: list[Input],
    outputs: list[Output],
    existing_components: list[str] | None = None,
    *,
    session: Session,
) -> tuple[list[ComponentResponsibilities], GapAnalysis]:
    """Loop: identify components -> analyze gaps -> refine until clean.

    Returns ComponentResponsibilities objects (not ComponentRequirement)
    with incidental notes captured from the user.
    """
    existing = existing_components or []
    max_iterations = 5

    for _ in range(max_iterations):
        requirements = ComponentRequirement.identify_from_chat(
            analysis=analysis,
            inputs=inputs,
            outputs=outputs,
            session=session,
        )
        gaps = _GapAnalysis.analyze_from_chat(
            components=requirements,
            analysis=analysis,
            existing_components=existing,
            session=session,
        )

        # Check if gaps are resolved
        if not gaps.missing_components and not gaps.integration_gaps and not gaps.organizational_gaps:
            # Convert to ComponentResponsibilities and capture incidental notes
            components = _requirements_to_responsibilities(requirements)
            components = _capture_incidental_notes(components, session=session)
            return components, gaps

        # Pass gap info back — the LLM sub-workflows will see it and adjust
        # by including gap context in the next iteration
        existing.extend(gaps.missing_components)

    # After max iterations, return best effort
    requirements = ComponentRequirement.identify_from_chat(
        analysis=analysis,
        inputs=inputs,
        outputs=outputs,
        session=session,
    )
    components = _requirements_to_responsibilities(requirements)
    components = _capture_incidental_notes(components, session=session)
    return components, _GapAnalysis.analyze_from_chat(
        components=requirements,
        analysis=analysis,
        existing_components=existing,
        session=session,
    )
