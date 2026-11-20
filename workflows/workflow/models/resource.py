# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import Session, atomic_workflow_step, composite_workflow_step
from chat_workflow.annotations import UserLabel
from chat_workflow.mixins import LLMValidated

from .deliverable import Deliverable
from .process_definition import ProcessDefinition

_NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the work for the user.",
)


class Resource(LLMValidated):
    """A single workflow resource."""

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
        UserLabel("where it comes from", "source"),
    ] = Field(
        ..., description="Where it comes from (a person, system, document, or place)", min_length=1
    )
    format: Annotated[
        str,
        UserLabel("what it looks like", "format"),
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
        UserLabel("what it needs first", "dependencies"),
    ] = Field(
        default_factory=list,
        description="Other things it needs first",
    )
    validation_criteria: Annotated[
        str,
        UserLabel(
            "how you know you have everything",
            "validation criteria",
            "criteria",
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

    @composite_workflow_step
    @classmethod
    def generate_from_chat(
        cls,
        deliverables: Annotated[list[Deliverable], "What the person makes"],
        analysis: Annotated[
            ProcessDefinition | None,
            "What the person has already told you about how their work happens",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
        session: Annotated[Session, "The chat session"] = None,  # type: ignore[assignment]
    ) -> list[Resource]:
        """Derive the typical starting materials, then confirm the list."""
        draft = _draft_resources(
            deliverables=deliverables,
            analysis=analysis,
            session=session,
        )
        session.say(_format_resources(draft, deliverables))
        return _confirm_resources(
            draft=draft, session=session, max_turns=max_turns
        )


def _format_resources(draft: list[Resource], deliverables: list[Deliverable]) -> str:
    """Present the typical starting materials in the user's words."""
    thing = deliverables[0].name if deliverables else "the thing you make"
    lines = [f"Let's list what you start with before you make {thing}.", ""]
    for resource in draft:
        line = f"- {resource.source} — form: {resource.format}"
        if resource.dependencies:
            line += f"; needs first: {'; '.join(resource.dependencies)}"
        if resource.validation_criteria:
            line += f"; the check: {resource.validation_criteria}"
        lines.append(line)
    lines.append("")
    lines.append("Which of these do you have, and what else do you start with?")
    return "\n".join(lines)


@atomic_workflow_step(chat=False, single_turn=True)
def _draft_resources(
    deliverables: Annotated[list[Deliverable], "What the person makes"],
    analysis: Annotated[
        ProcessDefinition | None,
        "What the person has already told you about how their work happens",
    ] = None,
    max_turns: Annotated[int, "Maximum turns"] = 1,
) -> list[Resource]:
    """Derive the things people usually start with for this kind of work.

The deliverables and any analysis are established. Propose the typical starting materials yourself; do not ask the
user anything. For each, in the user's words:
- where it comes from: the person, place, system, or document, not the act of getting it;
- what form it takes when it exists: a list, a document, a recording, not the act of gathering it;
- what it needs first, if anything;
- the check they apply: something true of the finished thing, not an action, an outcome, or a state of mind. Start
  the check with the thing, not with "you".
Use the deliverable's name; do not ask what to call it. Return the complete list."""
    ...  # type: ignore[reportReturnType]


@atomic_workflow_step
def _confirm_resources(
    draft: Annotated[
        list[Resource],
        "The complete list of typical starting materials already shown to the user",
    ],
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> list[Resource]:
    """The starting materials have been shown. Ask which ones they have.

If they add one, include it; if they say one does not apply, remove it. If they ask you to format or write the
deliverable itself, say in one sentence that this step lists what they start with and the deliverable comes next,
then return to the list. If they describe their domain content (specific dishes, articles, or ideas), acknowledge it in
one clause without praising it and return to the list of starting materials; never develop the content. Return the
corrected list when they confirm. Do not re-present the whole list and do not ask what to call the deliverable."""
    ...  # type: ignore[reportReturnType]
