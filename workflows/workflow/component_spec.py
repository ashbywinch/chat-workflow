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
        """Find the parts of the user's work.

    Each part is one thing: the thing itself, and the work of producing and changing it. A stage that merely drafts,
    reviews or sends another part's thing belongs to that part.

    Start with one sentence on what you are doing, then propose the parts you can already see, including the ones in
    `existing_components`. Name each after its thing ("Approval Record", not "Approval"). For each, say what it does end
    to end and at least one thing it does not do. Propose the remaining fields and the expertise you will act as;
    confirm them; do not ask for a job title. Describe what kind of thing it is in plain words — the thing it produces,
    or the plan or service it is. Never say the internal labels "value_stream", "artifact_producing", or
    "planning_service" aloud.

    Ask about gaps one at a time, in plain words, and call it "part of the work".

    When the user confirms the set, return it."""
        ...  # type: ignore[reportReturnType]


from .models.deliverable import Deliverable  # noqa: E402
from .models.process_definition import ProcessDefinition  # noqa: E402
from .models.resource import Resource  # noqa: E402
