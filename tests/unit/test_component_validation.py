"""Tests for Component and GeneratedComponent LLMValidated validation rules."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflows.workflow import GeneratedComponent
from workflows.workflow.component import Component


class TestComponentValidation(unittest.TestCase):
    def test_component_collects_metadata_rules(self):
        """Component validates its own metadata (name, purpose, expert_role)."""
        rules = Component.collect_all_rules()
        self.assertTrue(len(rules) >= 2)
        self.assertTrue(any("name" in r.lower() for r in rules),
                        "Should have a rule about the name field")

    def test_generated_component_collects_code_quality_rules(self):
        """GeneratedComponent validates the generated Python code quality."""
        rules = GeneratedComponent.collect_all_rules()
        self.assertTrue(len(rules) >= 4)
        self.assertTrue(any("BaseModel" in r for r in rules),
                        "Should validate that generated class inherits from BaseModel")
        self.assertTrue(any("Field" in r for r in rules),
                        "Should validate that fields have descriptions")
        self.assertTrue(any("min_length" in r for r in rules),
                        "Should validate min_length=1 on required fields")
        self.assertTrue(any("atomic_workflow" in r or "docstring" in r for r in rules),
                        "Should validate the @atomic_workflow method's docstring quality")

    def test_valid_component_passes_validation(self):
        """A properly constructed Component should not raise."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="Test",
                purpose="Test",
                code_path=p,
                model_class="TestModel",
                expert_role="Test Expert",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "Test")


if __name__ == "__main__":
    unittest.main(verbosity=2)