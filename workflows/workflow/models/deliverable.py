# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.confirmation import confirm_draft
from chat_workflow.conversation_rules import WARM_OPEN
from chat_workflow.mixins import LLMValidated
from chat_workflow.rendering import RenderContext, without_full_stop

from .process_definition import ProcessDefinition

_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the work for the user.",
)


class Deliverable(LLMValidated):
    """One thing the person makes."""

    model_config = ConfigDict(title="what you make")

    _validation_rules: ClassVar[list[str]] = [
        "The quality check is one someone can apply when the thing is "
        "produced — not an after-the-fact outcome like 'no one complains' or "
        "'everyone is happy'. A completeness check (the required fields are "
        "present) is valid when that is the user's standard, even for a "
        "personal record that may be saved with only a name.",
        "What happens to it next says who receives it and what they do with it, "
        "or that it stays with the user and is not handed on. Where it is kept "
        "says where. Neither restates what it contains.",
        "If the thing's name alone is ambiguous, the explanation says what it "
        "contains.",
    ]

    name: Annotated[
        str,
        UserLabel("what you call it", "name", guard=False),
    ] = Field(
        ...,
        description="What the person calls what they make, e.g. 'Meeting Minutes', 'Invoice PDF'",
        min_length=1,
    )
    description: Annotated[
        str | None,
        UserLabel("what it is", "description", guard=False),
    ] = Field(
        default=None,
        description="Plain-English description of what it contains — "
        "required unless the name alone is unambiguous (e.g. 'Invoice' is clear; "
        "'Widget' is not).",
    )
    consumer: Annotated[
        str,
        UserLabel("who it is for", "consumer"),
    ] = Field(
        ...,
        description="Who uses or receives it (a person, role, or team); name the party only — what they do with it belongs in integration_points",
        min_length=1,
    )
    format: Annotated[
        str,
        UserLabel("what it looks like", "format"),
    ] = Field(
        ...,
        description="What it looks like: its exact layout (e.g. 'one-page PDF', 'shared spreadsheet')",
        min_length=1,
    )
    success_criteria: Annotated[
        str,
        UserLabel("how you know it is good", "success criteria", "criteria"),
    ] = Field(
        ...,
        description=(
            "A check someone can apply when the thing is produced — not "
            "an after-the-fact outcome and not just its format"
        ),
        min_length=1,
    )
    integration_points: Annotated[
        str,
        UserLabel("who gets it next", "integration points"),
    ] = Field(
        ...,
        description="Who receives it next, and what they do with it",
        min_length=1,
    )
    storage_requirements: Annotated[
        str,
        UserLabel("where it is kept", "storage requirements"),
    ] = Field(
        ...,
        description="Where it is kept",
        min_length=1,
    )

    def render(self, context: RenderContext) -> str:
        """Say one thing the user makes, in the user's words.

        The document's own shape comes first, then its audience and the facts
        about how it is used, as sentences. Labels ("Complete when", "Where it
        is kept") read as sections of the document and as process jargon; a
        confused user does not parse them.
        """
        summary = f"{self.name} — {self.description}" if self.description else self.name
        lines = [
            summary,
            f"It looks like: {without_full_stop(self.format)}.",
            f"It's for {without_full_stop(self.consumer)}.",
            f"You'll know it's done when {without_full_stop(self.success_criteria)}.",
        ]
        if self.integration_points:
            lines.append(f"{without_full_stop(self.integration_points)}.")
        if self.storage_requirements:
            lines.append(f"{without_full_stop(self.storage_requirements)}.")
        return "\n".join(lines)

    @model_validator(mode="after")
    def _consumer_is_a_party(self) -> Deliverable:
        """Reject the artifact as its own consumer; the field says who it is for."""
        if self.consumer.strip().casefold() == self.name.strip().casefold():
            raise ValueError(
                "consumer must be a person, role, or team, not the thing itself"
            )
        return self

    @model_validator(mode="after")
    def _integration_is_not_the_consumer(self) -> Deliverable:
        """What happens next is not a repeat of who receives it."""
        if (
            self.integration_points.strip().casefold()
            == self.consumer.strip().casefold()
        ):
            raise ValueError(
                "integration_points must say what happens to it next — who "
                "receives it and what they do with it — not repeat the consumer"
            )
        return self

    @composite_workflow_step
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[
            ProcessDefinition | None,
            "What the person has already told you about how their work happens",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]
    ) -> Annotated[list[Deliverable], Field(min_length=1)]:
        """Gather what the user makes, propose its shape, then confirm it.

        The conversation describes the thing; the draft is derived from that
        description and from what people in that domain usually produce, then
        shown as a proposal the user corrects. A shape that is never shown is
        not a proposal, so this step never hands the user a form to fill in.
        """
        gathered = _gather_deliverable(
            analysis=analysis,
            session=session,
            max_turns=min(max_turns, 6),
        )
        draft = _draft_deliverables(
            notes=gathered.notes,
            analysis=analysis,
            session=session,
        )
        session.propose(draft)
        return confirm_draft(
            session=session,
            draft=draft,
            step=_confirm_deliverables,
            max_turns=min(max_turns, 4),
        )


class DeliverableNotes(BaseModel):
    """What the user said about the thing they make, in their own words."""

    notes: str = Field(
        min_length=1,
        description=(
            "The user's description of the thing they make, in their own "
            "words: what they call it, who it is for, what it contains, and "
            "what happens to it. Not your reply to them."
        ),
    )


@atomic_workflow_step(
    explore=True,
    # The opening asks what they make; their answer is what the notes hold.
    min_turns=2,
    conversation_validation_rules=[WARM_OPEN, _NO_EXECUTOR_MODE],
)
def _gather_deliverable(
    analysis: Annotated[
        ProcessDefinition | None,
        "What the person has already told you about how their work happens",
    ] = None,
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> DeliverableNotes:
    """Collect what the user makes, in their own words.

The conversation has already opened; do not greet again. Say in one sentence what you are doing — you help people describe what they end up with so it can be built. Then ask one question about the thing they make.

The thing may be personal — a meal plan, a trip plan, a letter — not only a work document. Do not refuse because it is not for a job. Never call it a "deliverable" or an "output": say the thing, the plan, the minutes, or the user's own word.

If the `analysis` or the conversation already names the thing they make, use that name and do not ask what they call it. Ask only what is still missing or uncertain — who it is for, what it contains, what happens to it next — one question at a time. If they name a common thing (meeting minutes, invoices), you may ask about its parts as someone who knows the kind: "Minutes usually have the date, who was there, the decisions, and action items with owners — does yours need all of those?" That is a question about their thing, not a finished proposal.

A private working copy the user never shares is not the thing they make; it is something they start with. When several things are produced, the last one is the thing the consumer receives; the others are its sections.

Stay at the level of the thing they make, not its contents. Do not ask them to paste their notes, read out what the thing contains, or share the actual material: you are describing the kind of thing they make, not making one. If they offer their notes or say "let's start with the notes", say the notes come later and return success now. Never praise their domain or develop its content; acknowledge an aside in one clause and return to the thing they make.

If they ask what it should look like, what headings it has, a format, or what you think, do not name any heading, section, or layout and do not ask another question: return success now. The next turn shows the complete proposed shape — who it is for, its format, the check it must pass, and where it goes — for them to correct. A structure named here is fragmented across turns and the complete proposal is lost. In your success message, say in one sentence that you will show them the shape you propose — do not say you have made it or present it yourself.

Put what they told you in `notes`, in their words — not your reply to them. Return only once the notes name the thing they make; if they have not said it, ask "What do you make?" and keep listening. If you still need an answer, ask it as a continue."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_deliverables(
    notes: Annotated[str, "What the person told you they make, in their words"],
    analysis: Annotated[
        ProcessDefinition | None,
        "What you already know about how their work happens",
    ] = None,
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> Annotated[list[Deliverable], Field(min_length=1)]:
    """Propose the concrete shape of what the user makes.

The notes are what the user told you; do not ask them anything. Derive the shape yourself and return one Deliverable per thing they make. Return the artifact they produce — the minutes, the plan, the invoice — not their working notes, their tools, or the meeting itself.

The `analysis` is optional context. When it is None, the notes alone are enough: derive the usual shape for the thing they named. Never return failure because the process analysis is absent.

Use their words for every part they covered. For every part they did not cover, propose what people in that domain usually use, and make it concrete: this is a proposal they will correct, so an empty or generic value is useless. If they named meeting minutes, propose what minutes contain (the date, who attended, the decisions, action items with an owner and a due date, and the next meeting if there is one), how they are laid out (a short structured document), who gets them, where they are kept, and a check someone can apply when the minutes exist (every decision has an owner; every action item names who does it and when).

In the user's words, for each thing:
- name: what they call it.
- description: what it contains, in one sentence; required unless the name alone is unambiguous.
- consumer: the person, role, or team it is for — never the thing itself.
- format: its exact layout, naming the parts it holds (for example "a one-page document with sections for attendees, decisions, and action items").
- success criteria: a check someone can apply the moment it exists — not an outcome like "everyone is happy", and not just its shape.
- integration points: who receives it next and what they do with it.
- storage requirements: where it is kept.

If every part is already in the notes, return what they said. Never return failure because the notes are thin: propose the usual shape for that kind of thing."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
def _confirm_deliverables(
    draft: Annotated[
        list[Deliverable],
        "The complete proposal already shown to the user",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> Annotated[list[Deliverable], Field(min_length=1)]:
    """Confirm or correct the proposal that was just shown.

The proposal was shown in the previous turn, and the user is answering it. Do not present it again and do not repeat the question they are answering.

If they ask a question about a part of it — the format, the check, who gets it, where it is kept — answer with one concrete recommendation in your message, then return the proposal with that part set to the recommendation. If they say "what do you think", give the recommendation itself, not another question.

If they correct or add a part, update it and return the corrected proposal; the framework shows the change and asks them to confirm it. If they confirm, return the proposal unchanged. If they say the thing is not what they make, replace it with what they say they make.

If they push back that this is not what they need — they ask for a method, a checklist, or how to do the work — do not treat it as confirmation and do not close. A compliment followed by "but" is not a yes. Say in one sentence that this step defines what they make and the method comes next, then give one concrete method idea in the same message (for example, "right after the meeting, pull decisions, actions, and next steps out of your notes before writing prose"). Return the proposal unchanged.

If they object to a line of the proposal — "I don't want the check / who it's for / where it's kept" — those lines are not sections of the thing: they are how you will know it is complete and where it lives. Say that in one sentence, change only what the thing itself needs (its format or what it contains), and if nothing about the thing changes, return the proposal unchanged and finish. Do not ask the same question again. If they say there is nothing to check, do not keep a condition with steps in it: set the check to the simplest true statement about the thing (for a list, that both sections are filled in), say it only says when the thing is done, and finish.

If they ask you to write the thing itself, say in one sentence that you are helping them decide what it looks like, then return to the proposal. If they describe their domain content, do not comment on it; say "Noted." and return. Never return failure because the user's notes lack detail: the proposal is what they correct."""
    ...  # type: ignore[reportReturnType]
