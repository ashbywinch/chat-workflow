from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.mixins import LLMValidated

from .process_definition import ProcessDefinition


@dataclass
class IntegrationGap:
    """An unclear handoff or incomplete interface between components."""

    between: str
    description: str


@dataclass
class OwnershipGap:
    """An activity that lacks clear ownership."""

    activity: str
    reason: str


class GapAnalysis(LLMValidated):
    """Analysis of missing elements, gaps, and recommendations."""

    _validation_rules: ClassVar[list[str]] = [
        "Entries in missing_components and missing_playbooks must reference "
        "specific domain concepts from the user's work, not "
        "generic labels (e.g. 'InvoiceManager', not 'missing component').",
        "Recommendations must be concrete, actionable next steps, not vague "
        "suggestions (e.g. 'Create an InvoiceManager component with input/output "
        "contracts', not 'address the gap').",
    ]

    missing_components: Annotated[
        list[str],
        UserLabel("parts that are missing", "missing components"),
    ] = Field(
        default_factory=list,
        description="Parts of the work that are named but not yet designed",
        min_length=0,
    )
    missing_playbooks: Annotated[
        list[str],
        UserLabel("steps that are not written down", "missing playbooks"),
    ] = Field(
        default_factory=list,
        description="Named sets of steps that are referred to but not described",
        min_length=0,
    )
    integration_gaps: Annotated[
        list[IntegrationGap],
        UserLabel("unclear handovers", "integration gaps"),
    ] = Field(
        default_factory=list,
        description="Where one part passes work to another and the connection is unclear",
    )
    organizational_gaps: Annotated[
        list[OwnershipGap],
        UserLabel("work with no clear owner", "organizational gaps"),
    ] = Field(
        default_factory=list,
        description="Work that has no clear owner",
    )
    recommendations: Annotated[
        list[str],
        UserLabel("what to do next", "recommendations"),
    ] = Field(
        default_factory=list,
        description="Concrete next steps that address each gap",
        min_length=0,
    )

    @atomic_workflow_step
    @classmethod
    def analyze_from_chat(
        cls,
        components: Annotated[list, "The parts identified so far"],
        analysis: Annotated[
            ProcessDefinition,
            "What the person has told you about how their work happens",
        ],
        existing_components: Annotated[
            list[str] | None, "Names of parts that already exist"
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> GapAnalysis:
        """You help someone spot what is missing from their work.

        Start from the parts and the picture they already described; do not
        ask them to repeat it. Do the check yourself; do not ask "are there any
        gaps?" as your only method.

        Check every step in the work against the parts. A step that acts on a
        thing an existing part already produces — reviewing, approving, sending
        the same document — is a step of that part, not a gap and not a new
        part. A gap is a step that produces its own thing (a distinct record or
        document) that no part produces. Name the missing part after that thing
        (for example a "Distribution Record" for a send that keeps its own
        log), not after the step. Also check unclear handovers between parts,
        and which ideas they mentioned have no part or step. If everything is
        covered, say so and return no gaps.

        Make one pass: name the gaps, summarize them, and ask the user to
        confirm. When they confirm, return success. Converge rather than
        exploring further; once the gaps are named, do not keep asking about
        new parts of the work. The user may believe nothing is missing; your
        job is to report what the work and the parts actually show.

        Sending is its own part only when it keeps its own record of what went
        out (a distribution log); otherwise it is a step of the part that owns
        the thing. Parsing and file handling are technical work: give them
        their own part only if they produce their own record.
        """
        ...  # type: ignore[reportReturnType]
