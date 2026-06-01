"""Tests for ComponentDesignSpec composite model."""
import unittest

from pydantic import ValidationError

from workflows.workflow.design_spec import ComponentDesignSpec
from workflows.workflow.domain_spec import ComponentDomainField, ComponentDomainSpec
from workflows.workflow.interaction_context import ComponentInteractionContext
from workflows.workflow.structure import ComponentStructure


class TestComponentDesignSpec(unittest.TestCase):
    """Construction and field validation tests for ComponentDesignSpec."""

    def make_valid_domain_spec(self) -> ComponentDomainSpec:
        return ComponentDomainSpec(
            name="meeting_minutes",
            description="Structured meeting minutes that capture what happened",
            fields=[
                ComponentDomainField(
                    name="action_owner",
                    domain_description="The person responsible",
                    field_type_hint="person name",
                ),
            ],
            what_good_looks_like=[
                "Attendees can immediately understand decisions made",
            ],
            expert_role="Meeting Coordinator",
        )

    def make_valid_structure(self) -> ComponentStructure:
        return ComponentStructure(description="Structured meeting minutes")

    def make_valid_interaction_context(self) -> ComponentInteractionContext:
        return ComponentInteractionContext(
            must_prioritize=["Always ask about decisions early"],
            auto_suggest=["Suggest action item owners"],
            user_pain_points=["Users often forget to list attendees"],
        )

    def make_valid(self, **overrides) -> ComponentDesignSpec:
        kwargs = dict(
            domain_spec=self.make_valid_domain_spec(),
            structure=self.make_valid_structure(),
            interaction_context=self.make_valid_interaction_context(),
        )
        kwargs.update(overrides)
        return ComponentDesignSpec(**kwargs)

    # --- Construction ---

    def test_valid_construction(self):
        spec = self.make_valid()
        self.assertIsInstance(spec.domain_spec, ComponentDomainSpec)
        self.assertIsInstance(spec.structure, ComponentStructure)
        self.assertIsInstance(spec.interaction_context, ComponentInteractionContext)

    def test_domain_spec_is_accessible(self):
        spec = self.make_valid()
        self.assertEqual(spec.domain_spec.name, "meeting_minutes")
        self.assertEqual(
            spec.domain_spec.description,
            "Structured meeting minutes that capture what happened",
        )

    def test_structure_is_accessible(self):
        spec = self.make_valid()
        self.assertEqual(
            spec.structure.description, "Structured meeting minutes"
        )

    def test_interaction_context_is_accessible(self):
        spec = self.make_valid()
        self.assertEqual(
            spec.interaction_context.must_prioritize,
            ["Always ask about decisions early"],
        )
        self.assertEqual(
            spec.interaction_context.auto_suggest,
            ["Suggest action item owners"],
        )
        self.assertEqual(
            spec.interaction_context.user_pain_points,
            ["Users often forget to list attendees"],
        )

    # --- Missing required fields ---

    def test_missing_domain_spec_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(domain_spec=None)

    def test_missing_structure_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(structure=None)

    def test_missing_interaction_context_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(interaction_context=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
