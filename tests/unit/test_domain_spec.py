"""Tests for ComponentDomainSpec and ComponentDomainField models."""

import unittest

from pydantic import ValidationError

from workflows.workflow.domain_spec import ComponentDomainField, ComponentDomainSpec


class TestComponentDomainField(unittest.TestCase):
    def test_construction_with_all_fields(self):
        field = ComponentDomainField(
            name="action owner",
            domain_description="The person responsible for completing this action item",
            field_type_hint="person name",
        )
        self.assertEqual(field.name, "action owner")
        self.assertEqual(
            field.domain_description,
            "The person responsible for completing this action item",
        )
        self.assertEqual(field.field_type_hint, "person name")

    def test_missing_required_fields_raises_error(self):
        with self.assertRaises(ValidationError):
            ComponentDomainField()

    def test_missing_name_raises_error(self):
        with self.assertRaises(ValidationError):
            ComponentDomainField(
                domain_description="some description",
                field_type_hint="some hint",
            )

    def test_domain_language_no_python_terminology(self):
        """Field descriptions must use domain language, not Python/Pydantic terms."""
        field = ComponentDomainField(
            name="decision",
            domain_description="The final choice made by the group during the meeting",
            field_type_hint="single decision statement",
        )
        description = field.domain_description.lower()
        self.assertNotIn("pydantic", description)
        self.assertNotIn("field", description)
        self.assertNotIn("class_docstring", description)
        self.assertNotIn("validation", description)


class TestComponentDomainSpec(unittest.TestCase):
    def make_valid_spec(self) -> ComponentDomainSpec:
        return ComponentDomainSpec(
            name="Meeting Minutes",
            description="Structured meeting minutes that capture what happened",
            fields=[
                ComponentDomainField(
                    name="action owner",
                    domain_description="The person responsible for completing this action item",
                    field_type_hint="person name",
                ),
                ComponentDomainField(
                    name="decision",
                    domain_description="The final choice made by the group",
                    field_type_hint="single decision statement",
                ),
            ],
            what_good_looks_like=[
                "Attendees can immediately understand decisions made",
                "Action items are clearly assigned with owners",
            ],
            expert_role="Meeting Minutes Administrator",
        )

    def test_construction_with_all_fields(self):
        spec = self.make_valid_spec()
        self.assertEqual(spec.name, "Meeting Minutes")
        self.assertEqual(
            spec.description,
            "Structured meeting minutes that capture what happened",
        )
        self.assertEqual(len(spec.fields), 2)
        self.assertEqual(spec.fields[0].name, "action owner")
        self.assertEqual(spec.fields[1].name, "decision")
        self.assertEqual(len(spec.what_good_looks_like), 2)
        self.assertEqual(
            spec.what_good_looks_like[0],
            "Attendees can immediately understand decisions made",
        )
        self.assertEqual(spec.expert_role, "Meeting Minutes Administrator")

    def test_what_good_looks_like_can_be_empty(self):
        spec = ComponentDomainSpec(
            name="Action Items",
            description="List of action items from a meeting",
            fields=[],
            what_good_looks_like=[],
            expert_role="Action Item Coordinator",
        )
        self.assertEqual(spec.what_good_looks_like, [])

    def test_fields_can_be_empty(self):
        spec = ComponentDomainSpec(
            name="Simple Note",
            description="A simple unstructured note",
            fields=[],
            what_good_looks_like=["Notes are clear and actionable"],
            expert_role="Note Taker",
        )
        self.assertEqual(spec.fields, [])

    def test_missing_required_fields_raises_error(self):
        with self.assertRaises(ValidationError):
            ComponentDomainSpec()

    def test_missing_name_raises_error(self):
        with self.assertRaises(ValidationError):
            ComponentDomainSpec(
                description="Some description",
                fields=[],
                what_good_looks_like=[],
                expert_role="Some Expert",
            )

    def test_domain_language_no_python_terminology(self):
        """Descriptions must use domain language, not Python/Pydantic terms."""
        spec = self.make_valid_spec()
        description = spec.description.lower()
        self.assertNotIn("pydantic", description)
        self.assertNotIn("field", description)
        self.assertNotIn("class_docstring", description)
        self.assertNotIn("validation", description)
        self.assertNotIn("base model", description)

    def test_name_matches_domain_convention(self):
        """Name should be a noun-based artifact name, matching ComponentResponsibilities convention."""
        spec = self.make_valid_spec()
        self.assertEqual(spec.name, "Meeting Minutes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
