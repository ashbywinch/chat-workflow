from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, ClassVar, override

from pydantic import Field, field_validator, model_validator

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
_BLOCK_LINE = re.compile(r"^(alt|opt)\b")

# Determiners and possessives do not change which party a name refers to.
_NAME_DETERMINERS = frozenset(
    {"the", "a", "an", "my", "our", "your", "their", "his", "her", "its", "this", "that"}
)


def _normalise_name(name: str) -> str:
    """Compare a diagram label with a record name without case/spacing/punctuation noise."""
    return re.sub(r"[\s_\-.,;:!?()/]+", " ", name.strip().casefold()).strip()


def _name_words(name: str) -> list[str]:
    """The words that identify a party, without determiners or possessives."""
    return [word for word in _normalise_name(name).split() if word not in _NAME_DETERMINERS]


def _matches_known(path: str, known: set[str]) -> bool:
    """A participant may shorten a record phrase, not invent a new one."""
    words = _name_words(path)
    for name in known:
        name_words = _name_words(name)
        if words == name_words:
            return True
        # A shortened party keeps at least two words when it is a prefix of
        # the record phrase.
        if len(words) >= 2 and name_words[: len(words)] == words:
            return True
        # A single head noun may shorten the phrase: "Team" for "My team",
        # "Attendees" for "Meeting attendees". Generic words are not parties.
        if (
            len(words) == 1
            and words[0] not in {"you", "i", "we", "they", "user", "system", "actor"}
            and name_words
            and name_words[-1] == words[0]
        ):
            return True
    return False


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
    arrows: list[tuple[str, str, str]] = []
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
            arrows.append((arrow.group(1), arrow.group(3), arrow.group(4)))

    if not participants:
        raise ValidationError(
            'The diagram declares no participants. Declare each one as '
            '`participant <Path> as "<Component Path>: <Playbook>"`.'
        )
    undeclared = sorted({name for source, target, _ in arrows for name in (source, target)} - set(participants))
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
    if not any(source != target for source, target, _ in arrows):
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
    for _, _, message in arrows:
        if len(message) > 60 and "<br/>" not in message:
            raise ValidationError(
                f"The message '{message[:60]}' is long; split it with `<br/>` "
                "at a natural break."
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
        if not _matches_known(path, known):
            raise ValidationError(
                f"The participant '{path}' is not a component, a component's "
                "owner, an input, or an output in this workflow. Use one of: "
                + ", ".join(sorted(known))
                + "."
            )

    pairs = {(source, target) for source, target, _ in arrows}
    consumers = {_normalise_name(output.consumer) for output in outputs}
    consumer_aliases = {
        alias
        for alias, label in participants.items()
        if any(
            _matches_known(label.split(":", 1)[0].strip(), {consumer})
            for consumer in consumers
        )
    }
    # A consumer receives the outcome; it is not a participant that exchanges
    # work with the component. A solo process plus its consumer needs no
    # artificial return arrow; a return flow is required only when more than
    # one work-exchanging participant remains.
    if len(set(participants) - consumer_aliases) > 1 and not any(
        (target, source) in pairs for source, target, _ in arrows
    ):
        raise ValidationError(
            "With more than one participant, the diagram needs a return flow: "
            "an arrow back to a participant who sent work earlier."
        )
    source, target, _ = arrows[-1]
    if source == target:
        raise ValidationError(
            "The last message is a self-loop. Send the outcome to the "
            "participant who receives the work, not back to the same one."
        )
    target_path = participants[target].split(":", 1)[0].strip()
    if consumers and any(
        _matches_known(target_path, {consumer}) for consumer in consumers
    ):
        return
    if (target, source) in pairs:
        raise ValidationError(
            "The last message is the return flow. Put the return flow before "
            "the outcome; the final message must be the send to the output's "
            "consumer"
            + (f" ({', '.join(sorted(consumers))})." if consumers else ".")
        )
    if consumers:
        raise ValidationError(
            f"The last message goes to '{target_path}', not to the output's "
            "consumer. Send the outcome to one of: "
            + ", ".join(sorted(consumers))
            + "."
        )


def _item_field(items: Any, field: str) -> list[str]:
    """Read one string field from a list of dicts or model objects."""
    values: list[str] = []
    for item in items or []:
        value = item.get(field) if isinstance(item, dict) else getattr(item, field, None)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _primary_output(outputs: Any, components: Any = None) -> tuple[str, str] | None:
    """The output the final arrow should carry, and its consumer.

    A private working copy — the notes the user writes for themselves, or a
    draft they never share — is not the outcome. Prefer an output whose
    consumer is not the user and is not the producing component's owner.
    """
    pairs = [
        (name, consumer)
        for name, consumer in zip(
            _item_field(outputs, "name"), _item_field(outputs, "consumer"), strict=False
        )
    ]
    if not pairs:
        return None
    owner_roles = {
        _normalise_name(owner)
        for owner in _item_field(components, "owner_role")
        if owner.strip()
    }
    self_words = {"you", "me", "i", "user", "self", "myself"}
    for name, consumer in pairs:
        words = set(_name_words(consumer.split(":", 1)[0]))
        if words & self_words:
            continue
        if _normalise_name(consumer.split(":", 1)[0]) in owner_roles:
            continue
        return name, consumer
    return pairs[0]


_GENERIC_PARTICIPANTS = frozenset(
    {"you", "i", "we", "they", "user", "system", "actor"}
)


def _repair_generic_participants(diagram: str, components: Any, outputs: Any) -> str:
    """Give a participant that is a pronoun a real component or consumer name.

    The model sometimes writes `participant You as "You: Note-taker"`. The
    contract requires a component or artifact path; the alias can stay, so
    only the label's path is replaced.
    """
    component_names = _item_field(components, "name")
    component = component_names[0] if component_names else "Component"
    lines: list[str] = []
    for line in diagram.splitlines():
        match = _PARTICIPANT_LINE.match(line.strip())
        if not match:
            lines.append(line)
            continue
        alias, label = match.group(1), match.group(2)
        path, _, playbook = label.partition(":")
        if path.strip().casefold() not in _GENERIC_PARTICIPANTS:
            lines.append(line)
            continue
        # A component or artifact path is always safe. The consumer can be
        # the user ("I", "You"), which would still fail the contract.
        replacement = component
        lines.append(
            f'    participant {alias} as "{replacement}: {playbook.strip() or "Receive"}"'
        )
    return "\n".join(lines)


def _repair_diagram_outcome(diagram: str, outputs: Any, components: Any = None) -> str:
    """Make the final message the outcome to the consumer.

    The record already names the consumer and the output, so the final arrow
    is mechanical. If the model ends on a self-loop or a return flow, append
    the outcome arrow instead of spending a retry on the shape.
    """
    primary = _primary_output(outputs, components)
    if primary is None:
        return diagram
    output_name, consumer = primary
    consumer_norm = _normalise_name(consumer)
    lines = diagram.splitlines()
    participants: dict[str, str] = {}
    for line in lines:
        match = _PARTICIPANT_LINE.match(line.strip())
        if match:
            participants[match.group(1)] = match.group(2)

    consumer_alias = next(
        (
            alias
            for alias, label in participants.items()
            if _matches_known(label.split(":", 1)[0].strip(), {consumer_norm})
        ),
        None,
    )
    if consumer_alias is None:
        consumer_alias = "Consumer"
        while consumer_alias in participants:
            consumer_alias += "X"
        lines.append(f'    participant {consumer_alias} as "{consumer}: Receive"')

    arrows = [
        (match.group(1), match.group(3))
        for line in lines
        if (match := _ARROW_LINE.match(line.strip()))
    ]
    if not arrows:
        return "\n".join(lines)
    source, target = arrows[-1]
    if target == consumer_alias and source != consumer_alias:
        return "\n".join(lines)

    producer = source if source != consumer_alias else target
    if producer == consumer_alias:
        producer = next(
            (alias for alias in participants if alias != consumer_alias),
            consumer_alias,
        )
    message = f"Send the final {output_name}"
    if len(message) > 60:
        message = "Send the final result"
    if lines and not lines[-1].strip():
        lines.pop()
    lines.append(f"    {producer}->>{consumer_alias}: {message}")
    return "\n".join(lines)


def _fallback_diagram(components: Any, outputs: Any) -> str:
    """A valid diagram from the record when the model returned no participants.

    The record names the component and the consumer; a diagram has to show the
    work, a decision point, a return flow and the outcome. The refine step
    gives the user the semantic correction; this keeps the mechanical contract
    from failing the whole run when the model returns only ``sequenceDiagram``.
    """
    component_names = _item_field(components, "name")
    primary = _primary_output(outputs, components)
    if not component_names or primary is None:
        return ""
    component = component_names[0]
    output_name, consumer = primary
    component_alias = re.sub(r"\W+", "", component.title()) or "Component"
    consumer_alias = re.sub(r"\W+", "", consumer.title()) or "Consumer"
    if consumer_alias == component_alias:
        consumer_alias += "Receiver"
    purpose = (_item_field(components, "purpose") or ["Do the work"])[0]
    work = purpose.split(".")[0].strip()[:50] or "Do the work"
    outcome = f"Send the final {output_name}"
    if len(outcome) > 60:
        outcome = "Send the final result"
    return "\n".join(
        [
            "sequenceDiagram",
            f'    participant {component_alias} as "{component}: Produce"',
            f'    participant {consumer_alias} as "{consumer}: Receive"',
            f"    {component_alias}->>{component_alias}: {work}",
            "    opt Review needed",
            f"        {component_alias}->>{consumer_alias}: Send the draft",
            f"        {consumer_alias}->>{component_alias}: Return corrections",
            "    end",
            f"    {component_alias}->>{consumer_alias}: {outcome}",
        ]
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
        Validation(
            "The diagram shows the process as a sequence with a clear hand-off and a final outcome"
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

    @model_validator(mode="before")
    @classmethod
    def _repair_the_outcome_arrow(cls, data: Any) -> Any:
        """Make the last diagram message the outcome to the consumer."""
        if isinstance(data, dict) and isinstance(data.get("diagram"), str):
            diagram = data["diagram"]
            if not any(
                line.strip().startswith("participant")
                for line in diagram.splitlines()
            ):
                fallback = _fallback_diagram(
                    data.get("components") or [], data.get("outputs") or []
                )
                if fallback:
                    diagram = fallback
            diagram = _repair_generic_participants(
                diagram,
                data.get("components") or [],
                data.get("outputs") or [],
            )
            repaired = _repair_diagram_outcome(
                diagram,
                data.get("outputs") or [],
                data.get("components") or [],
            )
            if repaired != data["diagram"]:
                data = {**data, "diagram": repaired}
        return data

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
        "All component names in `components` must be nouns",
        "A diagram participant label is `path: playbook`; the playbook "
        "(Produce, Receive) is not part of the component name and does not "
        "make the name a verb",
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
        - Shape (replace the names with the record's own):
          sequenceDiagram
              participant Minutes as "Meeting Minutes: Draft"
              participant Attendees as "Meeting attendees: Receive"
              Minutes->>Minutes: Draft the minutes
              opt Review needed
                  Minutes->>Attendees: Send the draft
                  Attendees->>Minutes: Return corrections
              end
              Minutes->>Attendees: Send the final minutes
          Use only a component, a component's owner, an input's source, an
          output's name, or an output's consumer as a participant. A step is
          an arrow, not a participant.
        - Declare every participant as `<Path> as "<Component Path>: <Playbook
          Name>"`: the path is one word with no spaces (`MeetingMinutes`, not
          `Meeting Minutes`); the quoted display name carries the spaces. Then a
          short label for the playbook it acts in (for example `participant
          MinutesDraft as "Minutes Draft: Draft"`). Never use an action sentence
          or a description of the component as the playbook name.
        - Show direct handoffs between components when the work simply passes
          from one to the next. Every participant must exchange at least one
          message with another participant: show the hand-off, and end with
          the outcome arrow to the output's consumer. Declare the consumer as
          a participant even when one person does the work; a diagram whose
          every arrow is a self-loop is invalid. Include a coordinator
          participant only when a component really does coordinate the others;
          do not invent one.
        - When the components list contains only artifact-producing components,
          use those components as the participants; do not add the owner role as
          a separate participant when it is a pronoun such as "You". The
          receiving participant is the output's consumer, copied from `outputs`;
          do not substitute a synonym. The work is the component's own step,
          the outcome goes to that consumer, and the return flow comes back to
          the component.
        - Show a decision point: use `alt` with an `else` branch for a choice
          between paths, or `opt` for a step that may or may not happen; an
          `opt` needs no else branch. If the description names no explicit
          choice, use an `opt` for a step the work implies could be skipped —
          a review, a correction, or an action item that may be absent — and
          say so in the user's words.
        - With more than one participant, show the return flow: an arrow back
          from the participant who received work to the one who sent it. When
          the second participant is the output's consumer, show their
          acknowledgement or feedback back to the component before the final
          send. A business role may be a participant,
          but do not give it a flow unless a component owns that activity; if
          no component owns the acknowledgement, show it as a step of the
          component that owns the artifact.
        - Use `<br/>` to split any long message across lines.
        - The final arrow is the send to the output's consumer. Put any
          acknowledgement or return flow before it; never end with a return
          arrow, a self-loop, or a message to the component's owner.
        - Show the entry point, every activity from the analysis, and a clear
          outcome. End with the outcome: the last arrow sends the finished
          artifact from the component that produced it to the output's
          consumer, copied from `outputs`. A `Note` is not a message and does
          not count as the outcome, and the last arrow must not be a self-loop.
        - Before returning, check every activity in `analysis.activities`
          against the diagram's message text. Each one must appear; add a
          missing activity as a self-loop on the component that owns it. Use
          the output's consumer for the final arrow, not a private working
          copy such as the user's own notes.

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
          not 'Distribution'; 'Approval Record', not 'Approval'; 'Review
          Notes', not 'Minutes Review'. A review that produces no record of
          its own is a step of the component that owns the minutes.
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
