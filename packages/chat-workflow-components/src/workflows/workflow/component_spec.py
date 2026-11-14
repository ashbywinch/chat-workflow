"""ComponentSpec model — defines what a component is responsible for."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field, field_validator

from chat_workflow import atomic_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.mixins import LLMValidated

# Activity nouns the model reaches for when it should name the artifact.
# A deterministic guard: the LLM judge handles the general case, but these
# exact words are the ones that repeat across retries.
_ABSTRACT_ACTIVITY_NAMES = frozenset(
    {
        "approval",
        "distribution",
        "review",
        "analysis",
        "planning",
        "tracking",
        "reporting",
        "processing",
    }
)


class ComponentSpec(LLMValidated):
    """The component spec: what a component is responsible for in the user's domain."""

    _validation_rules: ClassVar[list[str]] = [
        # The name is enforced by the `_name_is_a_concrete_artifact` field
        # validator: it is a structural check, not a domain-quality rule.
        # Q1 — overall responsibility. May span subcomponents, because those
        # are other components; it is one coherent reason to change.
        "The purpose states the end-to-end responsibility this component "
        "delivers: one coherent outcome for one group of people who would ask "
        "for it to change. It may include work this component delegates to "
        "other components; it must not combine unrelated responsibilities.",
        # Q2 — own responsibility. No subcomponents, existing or latent.
        "The scope description states what this component itself does, as "
        "opposed to other components. It describes exactly one artifact — the "
        "data members of the model this component defines — and the work that "
        "produces and changes it. Several operations on that artifact "
        "(creating, updating, reviewing, approving, sending it) are one "
        "artifact, not several. A piece of work that is itself an artifact — a section "
        "with its own content, audience, or quality criteria (for example a "
        "strategy section or a financial projection inside a business plan), a "
        "separate record, deliverable, or unrelated technical work such as "
        "parsing or file handling — is a latent subcomponent and does not "
        "belong here. It also states at least one thing it does not do (out of "
        "scope); naming the adjacent component's or another role's work is "
        "enough, and it does not need to enumerate unrelated activities. A "
        "workflow step on the same artifact (reviewing, approving, sending it) "
        "is in scope; do not treat it as a separate artifact merely because it "
        "could be a standalone deliverable — only a distinct record or "
        "deliverable of its own belongs in another component.",
        "The owner_role is the capacity the artifact belongs to — the role "
        "whose job includes it, which may be the user. Use 'Owner' only when "
        "the artifact is the owner's own work; if the owner acts as a "
        "maintenance manager for a sign-off sheet, the owner role is "
        "Maintenance Manager. A person who supplies the input is a valid "
        "owner when the artifact is their own work: a note-taker's own notes "
        "belong to the note-taker. They are not the owner only when someone "
        "else is accountable for the artifact. If the user does the work and "
        "is accountable, their role is valid even when another role "
        "traditionally owns that kind of artifact; do not replace the user's "
        "role with a domain assumption. Propose likely roles from domain "
        "knowledge and confirm them.",
    ]

    name: Annotated[
        str,
        UserLabel("what you call it", "name"),
    ] = Field(
        ...,
        description=(
            "Artifact-based name (concrete noun), e.g. 'Invoice' not "
            "'GenerateInvoice'; 'Approval Record' not 'Approval'"
        ),
        min_length=1,
    )
    purpose: Annotated[
        str,
        UserLabel("what it is for", "purpose"),
    ] = Field(
        ...,
        description=(
            "The end-to-end responsibility this component delivers — one "
            "coherent outcome for one group of people. It may include work "
            "delegated to other components."
        ),
        min_length=1,
    )
    scope_description: Annotated[
        str,
        UserLabel("what it covers", "scope description", "scope"),
    ] = Field(
        ...,
        description=(
            "What this component itself does, as opposed to other components. "
            "One artifact — the data members of the model it defines — and the "
            "operations that produce and change it; no piece of work that is "
            "itself an artifact (a section with its own content, audience, or "
            "quality), no separate record or deliverable, no unrelated "
            "technical work. Also states what is out of scope."
        ),
        min_length=1,
    )
    owner_role: Annotated[
        str,
        UserLabel("who it belongs to", "owner role", "owner"),
    ] = Field(
        ...,
        description=(
            "The capacity the artifact belongs to — the role whose job "
            "includes it. It may be the user. 'Owner' only when the artifact "
            "is the owner's own work; for example Maintenance Manager for a "
            "cleaning sign-off sheet."
        ),
        min_length=1,
    )
    required_inputs: Annotated[
        list[str],
        UserLabel("what it needs to start", "required inputs", "inputs"),
    ] = Field(
        ...,
        description=(
            "Names of inputs this component needs from other components or from outside the workflow"
        ),
    )
    component_type: Annotated[
        Literal["value_stream", "artifact_producing", "planning_service"],
        UserLabel(
            "what kind of thing it is",
            "component type",
            "artifact producing",
            "produces",
        ),
    ] = Field(
        ...,
        description=(
            "One of: value_stream (a continuous outcome or KPI someone is "
            "accountable for — director/C-level work, not a discrete document), "
            "artifact_producing (creates and updates a discrete artifact), "
            "planning_service (plans or coordinates work)"
        ),
    )
    incidental_notes: Annotated[
        str,
        UserLabel("anything else you mentioned", "incidental notes"),
    ] = Field(
        "",
        description=("Any incidental details the user mentioned about how the work is done"),
    )

    @field_validator("name")
    @classmethod
    def _name_is_a_concrete_artifact(cls, value: str) -> str:
        words = value.strip().casefold().split()
        if words and words[-1] in _ABSTRACT_ACTIVITY_NAMES:
            raise ValueError(
                f"'{value}' ends with the activity '{words[-1]}'. Name the "
                "artifact it produces, for example 'Minutes Distribution "
                "Record', 'Approval Record', 'Review Notes'."
            )
        return value

    @atomic_workflow_step
    @classmethod
    def identify_from_chat(
        cls,
        analysis: Annotated[
            ProcessDefinition,
            "What the person has told you about how their work happens",
        ],
        inputs: Annotated[list[Resource], "What the person starts with"],
        outputs: Annotated[list[Deliverable], "What the person makes"],
        existing_components: Annotated[
            list[str] | None,
            "Parts already identified; include them and add any that are missing",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[ComponentSpec]:
        """You help someone find the main components of their work.

        Use the language the user is using; never switch languages on your own.

        The user has described what they make, what they need to start, and
        how the work happens. Now help them identify the distinct components
        that make up the work. Start by saying in one sentence what you are
        doing, for example 'Let me check what I have understood so far.'

        The components in `existing_components` have already been identified.
        Include them in your proposal, and add any component that is needed but
        missing. They are what was said earlier, not a fixed list: if one turns
        out to be a stage of another component's artifact (drafted, reviewed,
        sent), merge it into that component and say so.

        IMPORTANT RULES:
        - Never use technical or framework terms with the user. Ask: "what are
          the main things that happen?" or "what are the distinct parts of
          this?" When you refer to the process analysis, say "the steps you
          take" or "the things you do"; never say "activities" or "phases".
        - Propose what you can based on what the user told you: "Based on what
          you've described, the main parts might be: the raw notes, the
          transcribed notes, the organized draft, the final minutes, and the
          follow-up items — does that capture it?"
        - Name each component after the concrete artifact it owns (an 'Approval
          Record', not 'Approval'; a 'Distribution Record', not
          'Distribution').
        - For each component, say what it is accountable for end to end, and
          what it covers and does not cover, in the user's words. Name at
          least one thing it does not do; naming the adjacent component's
          work is enough. Write a short paragraph per component; never a
          labelled list such as 'Name:', 'Purpose:', 'Type:'.
        - For each component, propose the role or capacity it belongs to,
          using domain knowledge: a cleaning sign-off sheet usually belongs to
          the Maintenance Manager or Facilities Lead; minutes to the meeting
          chair or organiser; a business plan to the Owner or Founder. The
          user themselves can hold it. Use 'Owner' only for the owner's own
          work, not when they act in a specialist capacity. The notes a
          note-taker writes are theirs; do not name the input supplier as the
          owner only when someone else owns the artifact (a scribe's notes
          become minutes owned by the meeting organizer). Offer two or three
          likely roles and let the user correct them.
        - A component may be responsible for the whole workflow by relying on
          the other components; its own scope is still the one thing it records.
        - The component that keeps the others moving may be one of them;
          include it if it does real work. A separate coordinator is optional:
          a simple flow may just hand each result to the next component.
        - If one component coordinates the others, it still owns its own
          artifact: the finished thing the other components' work feeds into —
          a plan, a report, the final minutes. It assembles and checks that
          artifact; it does not do their work. Taking notes, extracting
          actions, or writing a section belongs to the component that owns it.
          Sending is its own component when it keeps its own record of what
          went out; otherwise it is a workflow step on the artifact.
        - One component per artifact. The artifact is what the component
          records — the fields of the thing it produces. A piece of work that
          produces its own record (an approval, a distribution log, a tracker)
          belongs to its own component. A state of the same record (draft,
          reviewed, final) or a workflow step on it (create, update, approve,
          send) is not a new component: one component may have as many
          workflow steps on its one artifact as the work needs.
        - Converge: after one proposal and the user's corrections, return the
          components. Do not keep refining names or re-confirming the same list.
        - Every input the workflow starts with must appear in at least one
          component's `required_inputs`; the component that produces the first
          artifact needs it.
        """
        ...  # type: ignore[reportReturnType]


from .models.deliverable import Deliverable  # noqa: E402
from .models.process_definition import ProcessDefinition  # noqa: E402
from .models.resource import Resource  # noqa: E402
