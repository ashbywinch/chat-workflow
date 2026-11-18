from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, ClassVar, override

from pydantic import Field, field_validator

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import Blob, Validation
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import BlobSyncMixin, LLMValidated

from .component_spec import ComponentSpec
from .models import (
    Deliverable,
    GapAnalysis,
    ProcessDefinition,
    Resource,
    generate_from_chat,
)
from .models.gap_analysis import GapAnalysis as _GapAnalysis

_PARTICIPANT_LINE = re.compile(r'^participant\s+(\w+)\s+as\s+"([^"]+)"\s*$')
_ARROW_LINE = re.compile(r"^(\w+)\s*(-->>|->>|-->|->|--x|-x)\s*(\w+)\s*:\s*(.+)$")
_BLOCK_LINE = re.compile(r"^(alt|opt|loop)\b")


def _normalise_name(name: str) -> str:
    """Compare a diagram label with a record name without case/spacing noise."""
    return re.sub(r"[\s_-]+", " ", name.strip().casefold())


def _validate_diagram_structure(
    diagram: str,
    components: list[ComponentSpec],
    outputs: list[Deliverable],
    inputs: list[Resource],
) -> None:
    """Check the diagram's mechanical contract and name the repair.

    The model writes this diagram from the record. The rules below are facts
    about the text, so a wrong diagram fails here with the line to fix instead
    of reaching an LLM judge that can only say it is wrong. Semantic ownership
    stays with the judge; syntax, participants, flow, a decision point, the
    return flow and the outcome are decided here.
    """
    participants: dict[str, str] = {}
    arrows: list[tuple[str, str]] = []
    blocks: list[str] = []
    for line in diagram.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        participant = _PARTICIPANT_LINE.match(stripped)
        if participant:
            participants[participant.group(1)] = participant.group(2)
            continue
        if _BLOCK_LINE.match(stripped):
            blocks.append(stripped.split()[0])
            continue
        arrow = _ARROW_LINE.match(stripped)
        if arrow:
            arrows.append((arrow.group(1), arrow.group(3)))

    if not participants:
        raise ValidationError(
            'The diagram declares no participants. Declare each one as '
            '`participant <Path> as "<Component Path>: <Playbook>"`.'
        )
    undeclared = sorted({name for pair in arrows for name in pair} - set(participants))
    if undeclared:
        raise ValidationError(
            "The diagram uses participant(s) that are not declared: "
            + ", ".join(undeclared)
            + '. Declare each as `participant <Path> as "<Component Path>: <Playbook>"`.'
        )
    if not arrows:
        raise ValidationError(
            "The diagram has no messages. Show the work as arrows from one "
            "participant to another, starting at the entry point."
        )
    if not any(source != target for source, target in arrows):
        raise ValidationError(
            "Every message is a self-loop; the diagram shows no hand-off "
            "between participants. Show at least one message from one "
            "participant to another."
        )
    if not blocks:
        raise ValidationError(
            "The diagram has no decision point. Add an `alt` block with an "
            "`else` branch for a choice, or an `opt` block for a step that may "
            "or may not happen."
        )
    if len(participants) > 1 and not any((target, source) in arrows for source, target in arrows):
        raise ValidationError(
            "With more than one participant, the diagram needs a return flow: "
            "an arrow back to a participant who sent work earlier."
        )
    source, target = arrows[-1]
    if source == target:
        raise ValidationError(
            "The last message is a self-loop. Send the outcome to the "
            "participant who receives the work, not back to the same one."
        )

    known = (
        {_normalise_name(component.name) for component in components}
        | {_normalise_name(component.owner_role) for component in components}
        | {_normalise_name(output.name) for output in outputs}
        | {_normalise_name(output.consumer) for output in outputs}
        | {_normalise_name(resource.source) for resource in inputs}
    )
    for label in participants.values():
        path = label.split(":", 1)[0].strip()
        if _normalise_name(path) not in known:
            raise ValidationError(
                f"The participant '{path}' is not a component, a component's "
                "owner, an input, or an output in this workflow. Use one of: "
                + ", ".join(sorted(known))
                + "."
            )


class WorkflowSpec(BlobSyncMixin, LLMValidated):
    """A workflow specification.

    Describes a business process and the component specs it needs to be
    implemented end to end. This is a design artifact, not a runtime workflow.
    """

    # The CLI command is named for the artifact the method creates (a
    # workflow), not for this class. The spec is an internal term.
    __cli_name__ = "workflow"

    name: str = Field(
        ...,
        description="Workflow name ending in 'Workflow'",
        min_length=1,
    )

    diagram: Annotated[
        str,
        Blob(".mmd"),
        Validation("Must use <br/> tags for splitting long text across lines"),
        Validation(
            "Must show the entry point, the flow of work between participants, "
            "at least one decision point, a return flow when there is more than "
            "one participant, and a clear outcome. "
            "A decision point is either an `alt` block with `else` branches, "
            "for a choice between paths, or an `opt` block, for a step that "
            "may or may not happen; an `opt` needs no else branch. A return "
            "flow is an arrow back to a participant who sent work earlier. "
            "The outcome is the last message, to whoever receives the work. "
            "Work may pass directly from one component to another; a "
            "separate coordinator is optional and must not be invented just to "
            "have one"
        ),
        Validation("Every activity in the process should be represented in the sequence flow"),
        Validation(
            "Participants must be components, artifacts, or real business roles — "
            "not pronouns, placeholders, or generic labels such as 'System' or 'User'"
        ),
    ] = Field(
        ...,
        description=(
            "Mermaid sequence diagram as text (sequenceDiagram format). "
            "Participants are declared as `ComponentPath as \"Component Path: "
            "Playbook Name\"`: a noun for the component or artifact, then a short "
            "label for the playbook it acts in (for example `participant "
            "MinutesDraft as \"Minutes Draft: Draft\"`). Never use an action "
            "sentence or a description of the component as the playbook name."
        ),
        min_length=1,
    )

    @override
    def validate_deterministic_rules(self) -> None:
        """Run the diagram's mechanical contract before the LLM judge."""
        _validate_diagram_structure(self.diagram, self.components, self.outputs, self.inputs)

    @field_validator("diagram")
    @classmethod
    def _validate_diagram_syntax(cls, value: str) -> str:
        if "sequenceDiagram" not in value:
            raise ValueError("diagram must use sequenceDiagram format")
        directives = (line.strip() for line in value.splitlines())
        if any(line.startswith(("classDef", "style ")) for line in directives):
            raise ValueError("diagram must not include classDef or styling directives")

        # The participant naming contract is mechanical, so it is checked here
        # rather than by the LLM judge: `<Path> as "<Component Path>: <Playbook>"`.
        participant = re.compile(r'^\s*participant\s+\w+\s+as\s+"([^"]+)"\s*$')
        generic = {"you", "i", "we", "they", "user", "system", "actor"}
        for line in value.splitlines():
            if not line.strip().startswith("participant"):
                continue
            match = participant.match(line)
            if not match:
                raise ValueError(
                    'participant must use `<Path> as "<Component Path>: <Playbook>"`'
                )
            label = match.group(1)
            if ":" not in label:
                raise ValueError(
                    'participant label must be "<Component Path>: <Playbook>"'
                )
            component_path, playbook = (part.strip() for part in label.split(":", 1))
            if not component_path or not playbook:
                raise ValueError(
                    'participant label must name both a component and a playbook'
                )
            if component_path.casefold() in generic:
                raise ValueError(
                    "component path must name a component or artifact, not a "
                    "pronoun or generic label"
                )
            if len(playbook.split()) > 3:
                raise ValueError(
                    "playbook name must be a short label (one to three words), "
                    "not an action sentence"
                )
        return value

    inputs: list[Resource] = Field(
        ...,
        description=(
            "All workflow inputs with source, format, event-handling conditions, "
            "and validation analysis; may be empty when the workflow is started "
            "by a timer or an external event"
        ),
    )

    outputs: list[Deliverable] = Field(
        ...,
        description="All workflow outputs with consumer, format, success criteria, and integration analysis",
        min_length=1,
    )

    components: list[ComponentSpec] = Field(
        ...,
        description="Component specs required by this workflow",
        min_length=1,
    )

    trigger_conditions: str = Field(
        "",
        description=(
            "Events or conditions raised from inside the workflow step that "
            "invoke event-driven components — components called on a "
            "non-deterministic trigger rather than by deterministic control flow. "
            "Empty when every component is called deterministically. Not for "
            "external events that start the workflow."
        ),
    )

    gap_analysis: GapAnalysis | None = Field(
        None,
        description="Analysis of missing elements, integration gaps, and organizational gaps",
    )

    architectural_validation: str = Field(
        ...,
        description=(
            "Brief validation of ownership, alignment, separation of concerns, and dependency management"
        ),
        min_length=1,
    )

    _validation_rules: ClassVar[list[str]] = [
        "All component names must be nouns",
        "All component names must be concepts from the user's domain",
        "Every input must be consumed by a component that needs it to make decisions",
        "REJECT if any activity lacks proper component ownership",
    ]

    @atomic_workflow_step(chat=False)
    @classmethod
    def _generate_diagram(
        cls,
        analysis: Annotated[ProcessDefinition, "The process definition"],
        components: Annotated[list[ComponentSpec], "The identified components"],
        inputs: Annotated[list[Resource], "The workflow inputs"],
        outputs: Annotated[list[Deliverable], "The workflow outputs"],
        gap_analysis: Annotated[GapAnalysis | None, "Optional gap analysis to incorporate"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> WorkflowSpec:
        """No user conversation: generate the workflow record and diagram from
        the provided analysis.

        Diagram:
        - Start with `sequenceDiagram`.
        - Declare every participant as `<Path> as "<Component Path>: <Playbook
          Name>"`: a noun for the component or artifact, then a short label for
          the playbook it acts in (for example `participant MinutesDraft as
          "Minutes Draft: Draft"`). Never use an action sentence or a
          description of the component as the playbook name.
        - Show direct handoffs between components when the work simply passes
          from one to the next. Include a coordinator participant only when a
          component really does coordinate the others; do not invent one.
        - Use `alt` with an `else` branch for a choice between paths, or `opt`
          for a step that may or may not happen; an `opt` needs no else branch.
          Show the return flow as an arrow back to a participant who sent work
          earlier. A business role may be a participant,
          but do not give it a flow unless a component owns that activity; if
          no component owns the acknowledgement, show it as a step of the
          component that owns the artifact.
        - Use `<br/>` to split any long message across lines.
        - Show the entry point, every activity from the analysis, and a clear
          outcome.

        Record:
        - Copy names, purposes, scopes, and component types unchanged. Do not
          rewrite them.
        - If an input from `inputs` is missing from every component's
          `required_inputs`, add it to the component that consumes it. This is
          the only field you may change.
        - If an activity has no owning component, first check whether it acts
          on an artifact a component already owns (reviewing, approving,
          sending the same document). If so, it is a workflow step of that
          component. Add a component only when the activity produces its own
          artifact, and name it after that artifact: 'Distribution Record',
          not 'Distribution'; 'Approval Record', not 'Approval'.
        - Use only the activities listed in the analysis; do not add steps such
          as "receive the agenda" or "attend the meeting" unless they are in the
          analysis.
        - Every activity must be owned — by a component, or by a workflow step
          of the component that owns its artifact.
        - A coordinating participant owns the finished artifact assembled from
          the other participants' work; it does not own their activities. Do
          not put their work or sending into the coordinator's scope. Sending
          is its own part, even when the same person does it.
        - Every input in `inputs` must appear in at least one component's
          `required_inputs`.
        - If gap_analysis is null, leave it null and do not invent gaps.
        - architectural_validation should briefly cover ownership, alignment,
          separation of concerns, and dependencies.
        - Name the workflow after the main process or artifact and end with
          "Workflow".
        """
        ...  # type: ignore[reportReturnType]

    @classmethod
    def _create_diagram(
        cls,
        analysis: ProcessDefinition,
        components: list[ComponentSpec],
        inputs: list[Resource],
        outputs: list[Deliverable],
        gap_analysis: GapAnalysis | None = None,
        *,
        session: Session,
        max_refinements: int = 3,
    ) -> WorkflowSpec:
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
        workflow = _ensure_inputs_consumed(workflow, inputs)

        workflow.materialize_blobs(Path(tmp_dir))
        diagram_path = workflow.get_blob_path("diagram")
        if diagram_path:
            session.io.echo(f"Diagram saved to: {diagram_path}")

        for _ in range(max_refinements):
            # A rejected diagram costs turns: the model has to explain what it
            # is changing and ask when the user's answer left a rule unmet. Five
            # turns ran out mid-repair, so give the conversation room to finish.
            refined = refine(
                initial_object=workflow,
                max_turns=8,
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

    @composite_workflow_step
    @classmethod
    def create(
        cls,
        process_description: str = "",
        *,
        session: Session,
        max_refinements: int = 3,
        existing_components: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> WorkflowSpec:
        """Create a complete workflow artifact through conversation.

        Orchestrates: process analysis, component identification,
        gap resolution, diagram generation, user refinement, and
        component creation.

        ``output_dir`` is the parent directory for generated component files;
        defaults to ``workflows/<component name>/`` under the current directory.
        """
        session.io.echo("\nLet me understand how it all fits together.")
        analysis = generate_from_chat(
            session=session,
            process_description=process_description,
        )

        session.io.echo(
            f"\nGreat — {analysis.orchestrating_component} manages this process. Now, what does it produce?"
        )
        outputs = Deliverable.generate_from_chat(analysis=analysis, session=session)

        session.io.echo("\nNow let's figure out what you have to work with.")
        inputs = Resource.generate_from_chat(deliverables=outputs, session=session)

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
                    output_dir=output_dir,
                    deliverables=outputs,
                )
                session.io.echo(f"  ✓ {req.name} created at {component.code_path}")
            except Exception as e:
                session.io.echo(f"  ✗ Failed to create {req.name}: {e}")

        return workflow


def _normalize_component_spec(req: ComponentSpec) -> ComponentSpec:
    """Return the already-validated component spec unchanged."""
    return req


def _normalize_component_specs(
    requirements: list[ComponentSpec],
) -> list[ComponentSpec]:
    return [_normalize_component_spec(r) for r in requirements]


def _capture_incidental_notes(
    components: list[ComponentSpec],
    *,
    session: Session,
) -> list[ComponentSpec]:
    """Ask the user about incidental notes for each component.

    During WorkflowSpec conversations, the user may mention internal details,
    implementation preferences, or constraints about a component. This
    function gives them an explicit opportunity to share those notes.
    """
    result: list[ComponentSpec] = []
    for comp in components:
        session.io.echo(f"\nAny incidental notes about the '{comp.name}' component?")
        session.io.echo(
            "(e.g., internal details, implementation preferences, or constraints the component should follow)"
        )
        notes = session.io.prompt(f"Incidental notes for {comp.name}")
        result.append(comp.model_copy(update={"incidental_notes": notes}))
    return result


def _resolve_gaps(
    analysis: ProcessDefinition,
    inputs: list[Resource],
    outputs: list[Deliverable],
    existing_components: list[str] | None = None,
    *,
    session: Session,
) -> tuple[list[ComponentSpec], GapAnalysis]:
    """Loop: identify components -> analyze gaps -> refine until clean.

    Returns ComponentSpec objects with incidental notes captured from the user.
    """
    existing = existing_components or []
    max_iterations = 5

    for _ in range(max_iterations):
        requirements = ComponentSpec.identify_from_chat(
            analysis=analysis,
            inputs=inputs,
            outputs=outputs,
            existing_components=existing,
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
            # Convert to ComponentSpec and capture incidental notes
            components = _normalize_component_specs(requirements)
            components = _capture_incidental_notes(components, session=session)
            return components, gaps

        # Pass gap info back — the LLM sub-workflows will see it and adjust
        # by including gap context in the next iteration
        existing.extend(gaps.missing_components)

    # After max iterations, return best effort
    requirements = ComponentSpec.identify_from_chat(
        analysis=analysis,
        inputs=inputs,
        outputs=outputs,
        existing_components=existing,
        session=session,
    )
    components = _normalize_component_specs(requirements)
    components = _capture_incidental_notes(components, session=session)
    return components, _GapAnalysis.analyze_from_chat(
        components=requirements,
        analysis=analysis,
        existing_components=existing,
        session=session,
    )


def _ensure_inputs_consumed(
    workflow: WorkflowSpec, inputs: list[Resource]
) -> WorkflowSpec:
    """Attach every workflow input to at least one component.

    The diagram model is told to do this, but the invariant is mechanical:
    an unconsumed input belongs to the component that first works on it. Use
    the first artifact-producing component, or the first component when none
    is artifact-producing.
    """
    if not inputs or not workflow.components:
        return workflow
    consumed = {
        name.casefold()
        for component in workflow.components
        for name in component.required_inputs
    }
    target = next(
        (
            component
            for component in workflow.components
            if component.component_type == "artifact_producing"
        ),
        workflow.components[0],
    )
    for resource in inputs:
        if resource.source.casefold() in consumed or resource.format.casefold() in consumed:
            continue
        if resource.source not in target.required_inputs:
            target.required_inputs.append(resource.source)
        consumed.add(resource.source.casefold())
    return workflow
