"""Rendering a typed result for the person reading the conversation.

Every result record renders itself; the framework owns the surface, the
plain-text format, and the guardrail that stops schema vocabulary, Python
types, or markup from reaching a user.
"""

import json
import unittest
from typing import Annotated
from unittest.mock import patch

from pydantic import BaseModel, Field, TypeAdapter

from chat_workflow.models import AgentResponse
from chat_workflow.rendering import (
    RenderContext,
    RenderError,
    RenderSurface,
    plain_type,
    render_result,
    validate_rendering,
)
from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.domain_spec import (
    ComponentDomainField,
    ComponentDomainSpec,
)
from workflows.workflow.models.deliverable import Deliverable
from workflows.workflow.models.process_definition import (
    ProcessDefinition,
    ProcessStage,
)
from workflows.workflow.models.resource import Resource
from workflows.workflow.structure import (
    ComponentStructure,
    StructField,
    StructValidator,
)


def _deliverable() -> Deliverable:
    return Deliverable(
        name="Meeting Minutes",
        description="The written record of a meeting.",
        consumer="Meeting attendees",
        format="A shared document",
        success_criteria="Every decision has an owner.",
        integration_points="Attendees receive it by email.",
        storage_requirements="Shared drive",
    )


def _component() -> ComponentSpec:
    return ComponentSpec(
        name="Meeting Minutes",
        purpose="Write up what the meeting decided.",
        scope_description="Does not cover the agenda.",
        owner_role="You",
        required_inputs=["Raw notes"],
        component_type="artifact_producing",
        incidental_notes="",
    )


def _domain() -> ComponentDomainSpec:
    return ComponentDomainSpec(
        name="MeetingMinutes",
        description="The record of a meeting.",
        fields=[
            ComponentDomainField(
                name="meeting_date",
                domain_description="When it happened",
                field_type_hint="date",
            )
        ],
        what_good_looks_like=["A non-attendee can catch up"],
        expert_role="Note Taker",
    )


@patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
class TestRecordsRenderThemselves(unittest.TestCase):
    def test_a_process(self, _rules):
        text = render_result(
            ProcessDefinition(
                stages=[
                    ProcessStage(name="Capture", steps=["Scribble notes"]),
                    ProcessStage(name="Write", steps=["Type them up"]),
                ],
                orchestrating_component="You",
                participants=["You"],
            ),
            RenderContext(),
        )

        flat = " ".join(text.split())
        self.assertIn("The work has 2 stages", flat)
        self.assertIn("Stage 1 — Capture: Scribble notes", flat)
        self.assertIn("Stage 2 — Write: Type them up", flat)
        self.assertIn("You are the only person involved", flat)
        self.assertIn("You keep it moving", flat)

    def test_a_resource(self, _rules):
        text = render_result(
            Resource(
                source="Raw notes",
                format="Bullet points",
                dependencies=["The agenda"],
                validation_criteria="Every decision is captured.",
            ),
            RenderContext(),
        )

        self.assertIn("Raw notes — Bullet points", text)
        self.assertIn("Needs The agenda first", text)

    def test_a_deliverable(self, _rules):
        text = render_result(_deliverable(), RenderContext())

        self.assertIn("Meeting Minutes — The written record", text)
        self.assertIn("For Meeting attendees; A shared document", text)
        self.assertIn("Ready when", text)

    def test_a_component(self, _rules):
        text = render_result(_component(), RenderContext())

        self.assertIn("Meeting Minutes — Write up", text)
        self.assertIn("Covers: Does not cover the agenda", text)
        self.assertIn("Owned by: You", text)
        self.assertNotIn("component owns", text)

    def test_a_domain_spec(self, _rules):
        text = render_result(_domain(), RenderContext())

        self.assertIn("Fields: meeting_date — When it happened", text)
        self.assertIn("The role is Note Taker", text)

    def test_a_structure_uses_the_domain_hints(self, _rules):
        text = render_result(
            ComponentStructure(
                description="The artifact",
                fields=[
                    StructField(
                        name="meeting_date",
                        type_expr="str",
                        field_type_hint="date",
                        field_def_kwargs={"description": "When it happened"},
                    ),
                    StructField(
                        name="attendees",
                        type_expr="list[str]",
                        field_type_hint="list of person names",
                    ),
                ],
                model_validators=[
                    StructValidator(
                        rule="Every action item has an owner.",
                        domain_origin="A missed owner is a missed task.",
                    )
                ],
            ),
            RenderContext(),
        )

        self.assertIn("meeting date: date", text)
        self.assertIn("When it happened", text)
        self.assertIn("attendees: list of person names", text)
        self.assertIn("Rule: Every action item has an owner", text)

    def test_a_list_and_an_empty_list(self, _rules):
        text = render_result(
            [
                Resource(
                    source="Raw notes",
                    format="Bullet points",
                    dependencies=[],
                    validation_criteria="Every decision is captured.",
                ),
                Resource(
                    source="The agenda",
                    format="A document",
                    dependencies=[],
                    validation_criteria="Every item is covered.",
                ),
            ],
            RenderContext(),
        )

        self.assertIn("Raw notes", text)
        self.assertIn("The agenda", text)
        self.assertEqual(render_result([], RenderContext()), "")
        self.assertEqual(render_result(None, RenderContext()), "")


class TestTheGuardrail(unittest.TestCase):
    def _rejects(self, text: str, message: str, **context) -> None:
        with self.assertRaises(RenderError) as ctx:
            validate_rendering(text, RenderContext(**context))
        self.assertIn(message, str(ctx.exception))

    def test_the_users_own_words_are_never_censored(self):
        # A user's domain may legitimately use a word that is also a field
        # name; the guard checks display, not vocabulary.
        text = validate_rendering(
            "The consumer of this report is the finance team.",
            RenderContext(),
        )

        self.assertIn("consumer", text)

    def test_control_characters_and_markup_are_rejected(self):
        self._rejects("hello\x1b[31m", "control")
        self._rejects("<b>hello</b>", "HTML")
        self._rejects("**bold**", "markdown")
        self._rejects("# Heading", "headings")
        self._rejects("a | b", "table")

    def test_a_long_line_is_wrapped(self):
        text = validate_rendering(
            "Ready when " + " ".join(["specific"] * 30),
            RenderContext(max_line=60),
        )

        self.assertTrue(all(len(line) <= 60 for line in text.splitlines()))
        self.assertIn("\n", text)

    def test_an_unbreakable_word_is_rejected(self):
        self._rejects("x" * 30, "single word", max_line=20)

    def test_the_chat_is_the_default_surface(self):
        self.assertIs(RenderContext().surface, RenderSurface.CHAT)

    def test_plain_types_are_business_words(self):
        self.assertEqual(plain_type("str"), "text")
        self.assertEqual(plain_type("list[str | None]"), "a list of text")
        self.assertEqual(plain_type("list[ActionItem]"), "a list of action item")
        self.assertEqual(plain_type("dict[str, int]"), "a set of values")


class TestTheEmptyListDecision(unittest.TestCase):
    """The component decides; the framework only carries the schema."""

    class Item(BaseModel):
        name: str = Field(min_length=1)

    def test_a_plain_list_allows_empty(self):
        schema = json.dumps(TypeAdapter(AgentResponse[list[self.Item]]).json_schema())
        self.assertNotIn('"minItems"', schema)

    def test_an_annotated_list_requires_one(self):
        schema = json.dumps(
            TypeAdapter(
                AgentResponse[
                    Annotated[list[self.Item], Field(min_length=1)]
                ]
            ).json_schema()
        )
        self.assertIn('"minItems": 1', schema)

    def test_a_step_without_the_annotation_still_resolves(self):
        from chat_workflow.decorators import _inner_model

        self.assertIs(_inner_model(Annotated[list[Deliverable], Field(min_length=1)]), Deliverable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
