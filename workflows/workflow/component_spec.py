"""ComponentSpec model — defines what a component is responsible for."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, field_validator

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.confirmation import confirm_draft
from chat_workflow.mixins import LLMValidated
from chat_workflow.rendering import RenderContext
from chat_workflow.transcript import entries

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

    model_config = ConfigDict(title="a part of the work")

    _validation_rules: ClassVar[list[str]] = [
        # The name is enforced by the `_name_is_a_concrete_artifact` field
        # validator: it is a structural check, not a domain-quality rule.
        # Q1 — overall responsibility. May span subcomponents, because those
        # are other components; it is one coherent reason to change.
        "The overall responsibility states the end-to-end responsibility this part of the work "
        "delivers: one coherent outcome for one group of people who would ask "
        "for it to change. It may include work this component delegates to "
        "other components; it must not combine unrelated responsibilities.",
        # Q2 — own responsibility. No subcomponents, existing or latent.
        "The scope states what this part of the work itself does, as "
        "opposed to other components. It describes exactly one artifact — the "
        "data members of the model this component defines — and the work that "
        "produces and changes it. Several operations on that artifact "
        "(creating, updating, reviewing, approving, sending it) are one "
        "artifact, not several. A part of the work that is itself an artifact — a section "
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
        "The thing belongs to the capacity whose job includes it, which may "
        "be the user. Use 'the person it belongs to' only when the thing is "
        "that person's own work; for a maintenance manager's sign-off sheet, "
        "it is the maintenance manager. A person who supplies the input is "
        "valid when the thing is their own work: a note-taker's own notes "
        "belong to the note-taker. They are not the person only when someone "
        "else is accountable for the thing. If the user does the work and is "
        "accountable, that is valid even when another person traditionally "
        "keeps that kind of thing; do not replace the user with a domain "
        "assumption. Propose likely people from domain knowledge and confirm "
        "them.",
    ]

    name: Annotated[
        str,
        UserLabel("what you call it", "name", guard=False),
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
        UserLabel("what it is for", "purpose", guard=False),
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
            "operations that produce and change it; no part of the work that is "
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
            "One of: value_stream (someone is accountable for a continuous "
            "result or KPI), artifact_producing (it produces a document), "
            "planning_service (it plans or coordinates work). Say what it is "
            "in the user's words; the stored value stays in the result."
        ),
    )
    incidental_notes: Annotated[
        str,
        UserLabel("anything else you mentioned", "incidental notes"),
    ] = Field(
        "",
        description=("Any incidental details the user mentioned about how the work is done"),
    )

    def render(self, context: RenderContext) -> str:
        """Say one part of the work in the user's words."""
        lines = [f"{self.name} — {self.purpose}"]
        if self.scope_description:
            # The value may state what is *out* of scope ("Does not cover the
            # agenda"), so a "Covers:" label would contradict it.
            lines.append(f"What it does and doesn't cover: {self.scope_description}")
        if self.owner_role:
            lines.append(f"Owned by: {self.owner_role}")
        return "\n".join(lines)

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

    @field_validator("owner_role")
    @classmethod
    def _owner_is_a_capacity(cls, value: str) -> str:
        """A first-person owner is the user; say "You", not "I"."""
        if value.strip().casefold() in {"i", "me", "myself", "the user", "user"}:
            return "You"
        return value

    @composite_workflow_step
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
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]  # the step decorator injects the session
    ) -> Annotated[list[ComponentSpec], Field(min_length=1)]:
        """Derive the parts of the user's work, then confirm the set.

        The parts are derived from the analysis and the outputs, shown to the
        user in their words, and confirmed or corrected in one conversation.
        """
        draft = _draft_components(
            analysis=analysis,
            inputs=inputs,
            outputs=outputs,
            existing_components=existing_components,
            session=session,
        )
        already_shown = {name for name in existing_components or []}
        if already_shown and {part.name for part in draft} <= already_shown:
            # The gap resolution asks again with the set it already confirmed.
            # Showing and confirming the same parts a second time is repetition.
            return draft
        shown = [
            part
            for part in draft
            if part.name.strip().casefold()
            not in {name.strip().casefold() for name in already_shown}
        ]
        # The proposal is one turn, whether it is the whole set or only the
        # parts added to a set the user already confirmed.
        session.propose(
            shown or draft,
            changed=bool(already_shown),
            note="Everything else stays as we agreed." if already_shown else None,
        )
        confirmed = confirm_draft(
            session=session,
            draft=draft,
            step=_confirm_components,
            max_turns=min(max_turns, 6),
        )
        # The user may say there are no separate roles at all. That is a fact
        # about the work, not a new part: make them the owner and say so.
        if _says_self_only(_last_user_text(session)) and any(
            part.owner_role.strip().casefold() not in {"you", "me", "i"}
            for part in confirmed
        ):
            confirmed = [
                part.model_copy(update={"owner_role": "You"}) for part in confirmed
            ]
            session.say(
                "Got it — it's all you. I've made you the owner of every part."
            )
        return confirmed


def _last_user_text(session: Session) -> str:
    for kind, text in reversed(entries(session.state.messages)):
        if kind == "user":
            return text
    return ""


def _says_self_only(text: str) -> bool:
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "just me",
            "only me",
            "no separate roles",
            "i do it all",
            "i do everything",
            "that's me",
            "all me",
        )
    )


from .models.deliverable import Deliverable  # noqa: E402
from .models.process_definition import ProcessDefinition  # noqa: E402
from .models.resource import Resource  # noqa: E402


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_components(
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
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> Annotated[list[ComponentSpec], Field(min_length=1)]:
    """Derive the parts of the user's work from the analysis.

The analysis, inputs and outputs are established facts. The parts are derived from its stages and steps. The owner of
each part is the person or role whose job includes producing or maintaining that part.
`analysis.orchestrating_component` is the owner only when that role does the work; use
`analysis.participants` for a part another participant produces — a
note-taker's notes belong to the note-taker, while the minutes belong to the coordinator. If the analysis says "I" or
"me", use the role the notes or participants describe; if there is no role name, use "You". Never return failure because
the role is written as a pronoun, and never return failure because the analysis
does not name the parts directly: each stage is work on some part, and naming
that part is your job. Include the parts in `existing_components` and add any
that are missing. Do not ask the user anything: derive the complete set and
return it.

Each part is one thing: the thing itself, and the work of producing and changing it. A stage that merely drafts,
reviews or sends another part's thing belongs to that part. Name each part after its thing ("Approval Record", not
"Approval"). For each, say what it does end to end and at least one thing it does not do. Describe what kind of thing
it is in plain words — the thing it produces, or the plan or service it is. Never say the internal labels
"value_stream", "artifact_producing", or "planning_service"."""
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body


@atomic_workflow_step
def _confirm_components(
    draft: Annotated[
        list[ComponentSpec],
        "The complete set of parts already derived from the analysis",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> Annotated[list[ComponentSpec], Field(min_length=1)]:
    """
The parts were shown with one confirm-or-correct question, and the user is
answering it. Do not show the set again and do not repeat that question.

Do not re-present the whole set. When the user confirms, return the set in the same response; do not ask for a second
confirmation. If they correct a part, or add a requirement that belongs to an existing part, update that part and
return the corrected set; the framework shows the change. If the requirement is a new artifact, propose a new part. If
the user says the parts are not separate jobs, explain that those jobs are steps inside the one part and return the
set. If they ask about the flow, handoffs, or decision points, answer in one or two sentences in a continue message,
using the analysis' stages, and wait for their response. If they give new information, use a continue turn to propose
the new or updated parts; you are not out of turns. Ask only about a gap the draft cannot fill; do not ask the user to
name a part, say who owns it, or describe a stage the analysis already lists. Never say the schema words "consumer",
"owner_role", "scope_description", "purpose", or "component_type"; say "who it is for", "who owns it", "what it covers",
and "what it does".
    
    """
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body
