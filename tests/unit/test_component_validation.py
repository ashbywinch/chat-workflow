"""Tests for Component LLMValidated validation rules."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflows.workflow.component import Component


class TestComponentValidation(unittest.TestCase):
    def test_collects_validation_rules(self):
        """Component should have _validation_rules for its own metadata quality."""
        rules = Component.collect_all_rules()
        self.assertTrue(len(rules) >= 2)
        self.assertTrue(any("name" in r for r in rules),
                        "Should have a rule about the name field")
        self.assertTrue(any("purpose" in r or "expert_role" in r for r in rules),
                        "Should have rules about purpose or expert_role")

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
            # Should not raise
            self.assertEqual(c.name, "Test")


if __name__ == "__main__":
    unittest.main(verbosity=2)