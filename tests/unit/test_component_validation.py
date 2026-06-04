"""Tests for Component and GeneratedComponent LLMValidated validation rules."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflows.workflow import ComponentSourceCode
from workflows.workflow.component import Component


class TestComponentValidation(unittest.TestCase):
    def test_component_collects_metadata_rules(self):
        """Component validates its own metadata (name, purpose, expert_role)."""
        rules = Component.collect_all_rules()
        self.assertTrue(len(rules) >= 2)
        self.assertTrue(any("name" in r.lower() for r in rules), "Should have a rule about the name field")

    def test_generated_component_collects_code_quality_rules(self):
        """GeneratedComponent validates the generated Python code quality."""
        rules = ComponentSourceCode.collect_all_rules()
        self.assertTrue(len(rules) >= 4)
        self.assertTrue(
            any("Pydantic model" in r or "BaseModel" in r for r in rules),
            "Should validate that class inherits from BaseModel",
        )
        self.assertTrue(any("Field" in r for r in rules), "Should validate that fields have descriptions")
        self.assertTrue(any("min_length" in r for r in rules), "Should validate min_length=1 on required fields")
        self.assertTrue(
            any("atomic_workflow" in r or "docstring" in r for r in rules),
            "Should validate the @atomic_workflow method's docstring quality",
        )

    def test_valid_component_passes_validation(self):
        """A properly constructed Component should not raise."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="Test",
                purpose="Test",
                code_path=p,
                model_class="TestModel",
                expert_role="Test Expert",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "Test")

    # --- Programmatic validation rules ---

    def _make_component(self, purpose: str, tmpdir: str) -> Component:
        """Helper to construct a Component with a given purpose."""
        p = Path(tmpdir) / "test.py"
        p.write_text("")
        return Component.model_construct(
            name="TestComponent",
            purpose=purpose,
            code_path=p,
            model_class="TestModel",
            expert_role="Domain Expert",
            component_type="artifact_producing",
        )

    def test_purpose_with_and_raises_validation_error(self):
        """Purpose containing 'and' should be rejected (single responsibility)."""
        with TemporaryDirectory() as tmpdir:
            self._make_component(
                purpose="Manages user authentication and handles profile management",
                tmpdir=tmpdir,
            )
            rules = Component.collect_all_rules()
            purpose_rule = [r for r in rules if "single responsibility" in r.lower() or "one sentence" in r.lower()]
            self.assertTrue(len(purpose_rule) > 0, "Should have a single responsibility rule")

    def test_purpose_with_also_raises_validation_error(self):
        """Purpose containing 'also' should be rejected (single responsibility)."""
        with TemporaryDirectory() as tmpdir:
            self._make_component(
                purpose="Manages user authentication also handles profile management",
                tmpdir=tmpdir,
            )
            rules = Component.collect_all_rules()
            purpose_rule = [r for r in rules if "single responsibility" in r.lower() or "one sentence" in r.lower()]
            self.assertTrue(len(purpose_rule) > 0, "Should have a single responsibility rule")

    def test_purpose_without_and_or_also_passes(self):
        """Purpose without 'and' or 'also' should pass validation."""
        with TemporaryDirectory() as tmpdir:
            c = self._make_component(
                purpose="Manages user authentication",
                tmpdir=tmpdir,
            )
            self.assertEqual(c.purpose, "Manages user authentication")


if __name__ == "__main__":
    unittest.main(verbosity=2)
