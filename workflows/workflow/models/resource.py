# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.confirmation import confirm_draft
from chat_workflow.mixins import LLMValidated
from chat_workflow.rendering import RenderContext, without_full_stop

from .deliverable import Deliverable
from .process_definition import ProcessDefinition

# lucidlint: ignore-file global-state the no-executor rule is a prompt constant shared by the class and module-level steps
_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the work for the user.",
)


class Resource(LLMValidated):
    """One thing the person starts with.

    User facing label: "what you start with".
    """

    model_config = ConfigDict(title="what you start with")

    _validation_rules: ClassVar[list[str]] = [
        "Where it comes from names a person or group of people, a system, a "
        "document, or a place; what it looks like is described separately. A "
        "person or group is a valid origin (for example 'The organizer' with "
        "'meeting agenda'). If the thing is itself the material (for example "
        "handwritten notes), what it looks like must describe the same kind of "
        "material, not a contradictory one such as a database. A list the user "
        "keeps only in their head ('mental list', 'in my head') is a valid "
        "form when that is where they keep it.",
        "The check is one the user can apply to the finished thing — it says "
        "what must be there, not how they gather it, and not an after-the-fact "
        "outcome like 'no one complains'. A statement that every named part is "
        "represented in the finished thing ('every shelf and drawer has an "
        "entry') is a check on the thing, not an outcome.",
        "A thing must not depend on itself.",
    ]

    source: Annotated[
        str,
        UserLabel(phrase="where it comes from", aliases="source"),
    ] = Field(
        ..., description="Where it comes from (a person, system, document, or place)", min_length=1
    )
    format: Annotated[
        str,
        UserLabel(phrase="what it looks like", aliases="format"),
    ] = Field(
        ...,
        description=(
            "What it looks like when it exists (e.g. 'handwritten list', 'CSV "
            "export', 'email thread'), not the action used to gather it"
        ),
        min_length=1,
    )
    dependencies: Annotated[
        list[str],
        UserLabel(phrase="what it needs first", aliases="dependencies"),
    ] = Field(
        default_factory=list,
        description="Other things it needs first",
    )
    validation_criteria: Annotated[
        str,
        UserLabel(
            phrase="how you know you have everything",
            aliases=("validation criteria", "criteria"),
        ),
    ] = Field(
        ...,
        description=(
            "A check the user can apply when the resource is produced to know "
            "nothing is missing — not a restatement of how they gather it and "
            "not an after-the-fact outcome"
        ),
        min_length=1,
    )

    def render(self, context: RenderContext) -> str:
        """Say one thing the person starts with, in their words."""
        lines = [f"{self.source} — {self.format}"]
        if self.dependencies:
            lines.append(f"Needs {'; '.join(self.dependencies)} first.")
        lines.append(f"Ready when {without_full_stop(self.validation_criteria)}.")
        return "\n".join(lines)

    @composite_workflow_step
    @classmethod
    def generate_from_chat(
        cls,
        deliverables: Annotated[list[Deliverable], "What the person makes"],
        analysis: Annotated[
            ProcessDefinition | None,
            "What the person has already told you about how their work happens",
        ] = None,
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]  # the step decorator injects the session
    ) -> Annotated[list[Resource], Field(min_length=1)]:
        """Derive what people usually start with, then confirm the list."""
        draft = _draft_resources(
            deliverables=deliverables,
            analysis=analysis,
            session=session,
        )
        session.propose(draft)
        return confirm_draft(
            session=session,
            draft=draft,
            step=_confirm_resources,
        )


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_resources(
    deliverables: Annotated[list[Deliverable], "What the person makes"],
    analysis: Annotated[
        ProcessDefinition | None,
        "What the person has already told you about how their work happens",
    ] = None,
) -> Annotated[list[Resource], Field(min_length=1)]:
    """Derive the things people usually start with for this kind of work.

The deliverables and any analysis are established. A single deliverable is enough: use its name and description to
propose what people usually start with. A thing they start with is concrete — the notes
they took, the agenda, the attendee list, the ingredients they bought, a recording. Use what the deliverable and the
analysis say: what it is made from, and what the person would already have for that kind of work. It is not an activity
("the meetings I attend", "the meeting happening") and never the deliverable itself, the people it is for, or its
format: "the family" is who the plan is for, and "one-page plan" is the plan's format, not something you start with.
A template, a blank form, or the headings are the deliverable's format, not a material. A person is not the material: the
material is what they hand you (their notes, their list), and the person is where it comes from. A meeting is not a
material either; the notes or the agenda from it are. Use a noun phrase a person could hand you. Do not ask the user
anything and do not return
failure because there is no process description. For each, in the user's words:
- where it comes from: the person, place, system, or document, not the act of getting it;
- what form it takes when it exists: a list, a document, a recording, not the act of gathering it;
- what it needs first, if anything;
- the check they apply: something true of the finished thing, not an action, an outcome, or a state of mind. Start
  the check with the thing, not with "you".
Use the deliverable's name; do not ask what to call it. In your message, call these "the things you start with" or name them; never call them "materials". Never coin a label for a thing — "allowed foods list", "pantry-and-fridge list" — unless the user said it: use what they call it, or describe it in their words. Return the complete list."""
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body


@atomic_workflow_step
def _confirm_resources(
    draft: Annotated[
        list[Resource],
        "The complete list of things they start with, already shown to the user",
    ],
) -> Annotated[list[Resource], Field(min_length=1)]:
    """
The list of things they start with was shown with one confirm-or-correct question, and the user is answering it. Do not show the list again and do not repeat that question.

If they add one, include it; if they say one does not apply, remove it. If they say the list is wrong or they do not
have an item, replace it with what they say they do have; never return failure because the list needs correction. If
they ask you to format or write the deliverable itself, say in one sentence that this step lists what they start with
and the deliverable comes next, then return to the list. If they ask about a different process, or say they are not
talking about this deliverable, say in one sentence that this step lists what they start with for that deliverable and
return the list. Do not return failure because the question is off-topic or because they are confused about the
process name. If they describe domain content, say "Noted." and return to the list. If they ask what you think or for
a suggestion — especially about a material's format or completeness — set that material's format and check to one
concrete choice and give the choice itself in your message, then return the corrected list. A question about a
starting item's own format or completeness — are the notes bullets, sentences, or keywords, and what makes them
ready to use — is a question about that item: answer with one concrete format and one concrete check, set them on the
item, and return the corrected list. It is not a question about the minutes, and it is not a reason to leave the list
unchanged. If they say "you tell me" or ask for suggestions, give one concrete choice for the material and update
it; never thank them and finish without a choice. Do not say "here is what I would suggest" and describe the step instead of the material. If their
feedback is about the minutes themselves — a field the minutes should have, or what an action item must carry — say in
one sentence that those belong to the minutes you already defined and this step lists what they start with, then return
the list unchanged. Never announce the list without giving it.
If they confirm, return the list; if they correct it, return the corrected list and the framework shows the change. Do
not re-present the whole list and do not ask what to call the deliverable.
    
    """
    ...  # type: ignore[reportReturnType]  # the step decorator supplies the body
