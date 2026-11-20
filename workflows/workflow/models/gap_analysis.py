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
    """Where one step passes to the next and the connection is unclear."""

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
        "Where something is missing, it names a specific domain concept from "
        "the user's work, not a generic label (e.g. 'InvoiceManager', not "
        "'missing part'). When nothing is missing, the lists are empty and "
        "that is valid.",
        "The next steps, when there are gaps, are concrete and actionable, not "
        "vague suggestions (e.g. 'Create an InvoiceManager part with its inputs "
        "and outputs', not 'address the gap'). When nothing is missing, the "
        "recommendations list is empty and that is valid.",
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
        UserLabel("connections between steps", "integration gaps"),
    ] = Field(
        default_factory=list,
        description="Where one part passes work to another and the connection is unclear",
    )
    organizational_gaps: Annotated[
        list[OwnershipGap],
        UserLabel("work with no one in charge", "organizational gaps"),
    ] = Field(
        default_factory=list,
        description="Work that no one has taken responsibility for",
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

        Say in one sentence what you are checking. Report only what is missing,
        one gap at a time, and ask the user to confirm each one. Watch for two
        answers: if they say a gap is not real, drop it; if they correct or add
        one, use their words. When they confirm — or when nothing is missing —
        say so in one sentence and return; do not ask about new parts of the
        work.

        The parts and the picture are already given. Do not introduce them,
        restate them, or read them back — the user gave you that. Compare the
        steps with the parts yourself and report only what is missing. If
        nothing is missing, return empty lists, including recommendations.

        Check every step in the work against the parts. A step that acts on a
        thing an existing part already produces — reviewing, approving, sending
        the same document — is a step of that part, not something missing and
        not a new part. Something is missing only when a step produces its own
        thing (a distinct written record) that no part produces. Name the
        missing part after that thing (for example a "Distribution Record" for
        a send that keeps its own note), not after the step. Also check where
        one step passes to the next, and which ideas they mentioned have no
        part or step.

        Converge rather than exploring further; once the missing parts are
        named, do not keep asking about new parts of the work. The user may
        believe nothing is missing; your job is to report what the work and the
        parts actually show. Sending is its own part only when it keeps its own
        written record of what went out; otherwise it is a step of the part
        that produces the thing. Parsing and file handling are technical work:
        give them their own part only if they produce their own record.
        """
        ...  # type: ignore[reportReturnType]
