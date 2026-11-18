# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
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
        "document, or a place; its "
        "shape is described separately. A person or group is a valid origin "
        "(for example 'The organizer' with shape 'meeting agenda'). If the "
        "thing is itself the material (for example handwritten notes), its "
        "shape must describe the same kind of material, not a contradictory "
        "one such as a database.",
        "The check is one the user can apply to the finished thing — it says "
        "what must be there, not how they gather it, and not an after-the-fact "
        "outcome like 'no one complains'.",
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
            "Its exact shape when it exists (e.g. 'handwritten list', 'CSV "
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

    @atomic_workflow_step(conversation_validation_rules=[_NO_EXECUTOR_MODE])
    @classmethod
    def generate_from_chat(
        cls,
        deliverables: Annotated[list[Deliverable], "What the person makes"],
        analysis: Annotated[
            ProcessDefinition | None,
            "What the person has already told you about how their work happens",
        ] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Resource]:
        """List what the user starts with before they make the thing they make.

    Open with one sentence on what you are doing, name the output from `deliverables`, and propose the things people
    usually start with for that kind of work; ask which of them they have.

    For each one, in the user's words:
    - where it comes from: a person, place, system, or document, not the act of getting it ("The fridge and freezer",
    not "Looking in the fridge"). If they know it from experience, the source is the person (them), and its shape says
    it is in their head; never make the experience itself the source;
    - what it looks like when it exists: a list, a document, a recording, not the act of gathering it ("Verbal
    feedback", not "Chatting to the sous");
    - what it needs first, if anything;
    - the check they apply: something true of the finished thing, not
    an action, not an outcome, and never a state of mind. "The list includes
    every item on every shelf, flagged if it is near expiry" - not "I know what
    is on every shelf", not "I can list every item", not "Every shelf has been
    checked", not "No one complains". Name what must be present, not a general
    description of the contents: "Every day of the week has an entry" is a
    check; "It contains a typical number of days" is a description. If they
    answer with a state of mind or an outcome, propose the concrete check
    yourself and ask them to confirm it.

    Work through one at a time: propose its shape and check, ask the user to confirm or correct, and only then move to
    the next. When they name one, accept it; do not read back a longer list of alternatives. Do not put checks for
    several in one message. Raw notes, and anything the user trusts as sent, still get a
    minimal check on the thing itself; do not ask how they know their own notes are complete. If they have only one, do
    not ask for more.

    Ask once and accept the answer. When they ask for a suggestion, give one concrete option in that same reply. When
    they confirm the set, return it without summarising it again.

    Describe each in the user's words. Say "what you start with". Stay at the
    level of what they start with, not its contents: if they describe their
    domain, acknowledge it in a clause and return to them."""
        ...  # type: ignore[reportReturnType]
