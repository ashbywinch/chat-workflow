"""Tests for Component and ComponentSourceCode LLMValidated validation rules."""

import unittest
from pathlib import Path

from workflows.workflow import ComponentSourceCode
from workflows.workflow.component import Component


class TestComponentValidation(unittest.TestCase):
    def test_component_collects_metadata_rules(self):
        """Component validates its own metadata (name, purpose, expert_role)."""
        rules = Component.collect_all_rules()
        self.assertTrue(len(rules) >= 2)
        self.assertTrue(any("name" in r.lower() for r in rules), "Should have a rule about the name field")

    def test_generated_component_collects_code_quality_rules(self):
        """ComponentSourceCode judges the judgment calls, not schema mechanics."""
        rules = ComponentSourceCode.collect_all_rules()
        self.assertTrue(len(rules) >= 4)
        self.assertTrue(
            any("LLMValidated" in r or "primary model" in r for r in rules),
            "Should validate the primary model's self-enforced quality rules",
        )
        self.assertTrue(
            any("domain-quality" in r for r in rules),
            "Should reserve generated-model rules for domain quality",
        )
        self.assertFalse(
            any("description=" in r for r in rules),
            "Field descriptions are deterministic, not judged",
        )
        self.assertFalse(
            any("min_length" in r for r in rules),
            "min_length is deterministic, not judged",
        )
        self.assertTrue(
            any("atomic_workflow_step" in r or "docstring" in r for r in rules),
            "Should validate the @atomic_workflow_step method's docstring quality",
        )

    def test_valid_component_passes_validation(self):
        """A properly constructed Component should not raise."""
        c = Component.model_construct(
            name="Test",
            purpose="Test",
            code_path=Path("test.py"),
            model_class="TestModel",
            expert_role="Test Expert",
            component_type="artifact_producing",
            owner_role="Owner",
        )
        self.assertEqual(c.name, "Test")

    # --- Programmatic validation rules ---

    def _make_component(self, purpose: str) -> Component:
        """Helper to construct a Component with a given purpose."""
        return Component.model_construct(
            name="TestComponent",
            purpose=purpose,
            code_path=Path("test.py"),
            model_class="TestModel",
            expert_role="Domain Expert",
            component_type="artifact_producing",
            owner_role="Owner",
        )

    def test_purpose_with_and_has_a_single_responsibility_rule(self):
        """Purpose containing 'and' should be rejected (single responsibility)."""
        self._make_component(
            purpose="Manages user authentication and handles profile management",
        )
        rules = Component.collect_all_rules()
        purpose_rule = [r for r in rules if "single responsibility" in r.lower() or "one sentence" in r.lower()]
        self.assertTrue(len(purpose_rule) > 0, "Should have a single responsibility rule")

    def test_purpose_with_also_has_a_single_responsibility_rule(self):
        """Purpose containing 'also' should be rejected (single responsibility)."""
        self._make_component(
            purpose="Manages user authentication also handles profile management",
        )
        rules = Component.collect_all_rules()
        purpose_rule = [r for r in rules if "single responsibility" in r.lower() or "one sentence" in r.lower()]
        self.assertTrue(len(purpose_rule) > 0, "Should have a single responsibility rule")

    def test_purpose_without_and_or_also_passes(self):
        """Purpose without 'and' or 'also' should pass validation."""
        c = self._make_component(purpose="Manages user authentication")
        self.assertEqual(c.purpose, "Manages user authentication")


if __name__ == "__main__":
    unittest.main(verbosity=2)
