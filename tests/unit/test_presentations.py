"""The deterministic presentations of derived workflow values.

Each formatter is what the user reads; it must say the value in their words,
without schema field names, and must keep the domain's own type hints.
"""

import unittest
from unittest.mock import patch

from workflows.workflow.component_spec import ComponentSpec, _format_components
from workflows.workflow.domain_spec import (
    ComponentDomainField,
    ComponentDomainSpec,
    _format_domain_spec,
)
from workflows.workflow.interaction_context import (
    ComponentInteractionContext,
    _format_interaction_context,
)
from workflows.workflow.models.deliverable import Deliverable
from workflows.workflow.models.process_definition import (
    ProcessDefinition,
    _format_process_definition,
)
from workflows.workflow.models.resource import Resource, _format_resources
from workflows.workflow.structure import (
    ComponentStructure,
    StructField,
    StructValidator,
    _format_structure,
    _plain_type,
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


@patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
class TestPresentations(unittest.TestCase):
    def test_a_process_reads_as_a_picture(self, _rules):
        text = _format_process_definition(
            ProcessDefinition(
                phases=["Capture", "Write"],
                activities=["Scribble notes", "Type them up"],
                orchestrating_component="You",
                participants=["You"],
            )
        )

        self.assertIn("Stages: Capture → Write", text)
        self.assertIn("Steps: Scribble notes; Type them up", text)
        self.assertIn("In charge: You", text)
        self.assertNotIn("phases", text)

    def test_resources_name_their_source_and_check(self, _rules):
        text = _format_resources(
            [
                Resource(
                    source="Raw notes",
                    format="Bullet points",
                    dependencies=["The agenda"],
                    validation_criteria="Every decision is captured.",
                )
            ],
            [_deliverable()],
        )

        self.assertIn("before you make Meeting Minutes", text)
        self.assertIn("which is Bullet points", text)
        self.assertIn("you need The agenda first", text)
        self.assertIn("Every decision is captured", text)
        self.assertIn("Which of these do you have", text)

    def test_a_component_is_responsible_for_its_purpose(self, _rules):
        text = _format_components([_component()])

        self.assertIn("This part is responsible for", text)
        self.assertIn("belongs to You", text)
        self.assertNotIn("component owns", text)

    def test_a_second_component_pass_shows_only_the_change(self, _rules):
        text = _format_components([_component()], existing=["Meeting Minutes"])

        self.assertEqual(text, "")

        changed = _format_components(
            [_component()], existing=["Rough Notes"]
        )
        self.assertIn("here's what I'd add or change", changed)
        self.assertIn("Everything else stays as we agreed", changed)

    def test_a_domain_spec_uses_the_artifact_name(self, _rules):
        text = _format_domain_spec(
            ComponentDomainSpec(
                name="MeetingMinutes",
                description="The record of a meeting.",
                fields=[
                    ComponentDomainField(
                        name="decision",
                        domain_description="What was agreed",
                        field_type_hint="text",
                    )
                ],
                what_good_looks_like=["A non-attendee can catch up"],
                expert_role="Note Taker",
            )
        )

        self.assertIn("MeetingMinutes", text)
        self.assertIn("decision — What was agreed", text)
        self.assertIn("A non-attendee can catch up", text)

    def test_a_structure_keeps_the_domain_hint(self, _rules):
        spec = ComponentDomainSpec(
            name="MeetingMinutes",
            description="The record of a meeting.",
            fields=[
                ComponentDomainField(
                    name="meeting_date",
                    domain_description="When it happened",
                    field_type_hint="date",
                ),
                ComponentDomainField(
                    name="attendees",
                    domain_description="Who was there",
                    field_type_hint="list of person names",
                ),
            ],
            what_good_looks_like=["It is skimmable"],
            expert_role="Note Taker",
        )
        text = _format_structure(
            ComponentStructure(
                description="The artifact",
                fields=[
                    StructField(name="meeting_date", type_expr="str"),
                    StructField(name="attendees", type_expr="list[str]"),
                ],
                model_validators=[
                    StructValidator(
                        rule="Every action item has an owner.",
                        domain_origin="A missed owner is a missed task.",
                    )
                ],
            ),
            spec,
        )

        self.assertIn("meeting date: date", text)
        self.assertIn("attendees: list of person names", text)
        self.assertIn("Every action item has an owner", text)

    def test_plain_types_are_business_words(self, _rules):
        self.assertEqual(_plain_type("str"), "text")
        self.assertEqual(_plain_type("list[str | None]"), "a list of text")
        self.assertEqual(_plain_type("list[ActionItem]"), "a list of action item")
        self.assertEqual(_plain_type("dict[str, int]"), "a set of values")

    def test_an_interaction_proposal_carries_the_questions(self, _rules):
        text = _format_interaction_context(
            ComponentInteractionContext(
                must_prioritize=["Decisions"],
                auto_suggest=["Owners"],
                tone_preference="Direct",
                user_pain_points=["Missing attendees"],
            )
        )

        self.assertIn("I'll always cover: Decisions", text)
        self.assertIn("I'll suggest: Owners", text)
        self.assertIn("Tone: Direct", text)
        self.assertIn("I'll watch for: Missing attendees", text)
        self.assertIn("What would you add", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
