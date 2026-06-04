"""Unit tests for GeneratedComponent programmatic validation rules.

Tests the @model_validator(mode="after") on GeneratedComponent that checks:
- Valid Python syntax
- Exactly one BaseModel subclass
- @atomic_workflow decorator present

Also tests that new _validation_rules entries are collected.
"""

import unittest
from unittest.mock import patch

from tests.sample_code import (
    INVALID_SYNTAX_CODE,
    MULTIPLE_BASEMODEL_CODE,
    NO_BASEMODEL_CODE,
    NO_WORKFLOW_CODE,
    VALID_COMPONENT_CODE,
)
from workflows.workflow.component_source_code import ComponentSourceCode


class TestGeneratedComponentProgrammaticValidation(unittest.TestCase):
    """Tests for the programmatic @model_validator on GeneratedComponent."""

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_valid_code_passes_validation(self, mock_collect):
        """Valid code with one BaseModel and @atomic_workflow should pass."""
        component = ComponentSourceCode(code=VALID_COMPONENT_CODE)
        self.assertIn("class MinutesDraft", component.code)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_invalid_syntax_raises_validation_error(self, mock_collect):
        """Invalid Python syntax should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            ComponentSourceCode(code=INVALID_SYNTAX_CODE)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_no_basemodel_raises_validation_error(self, mock_collect):
        """Code with no BaseModel subclass should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            ComponentSourceCode(code=NO_BASEMODEL_CODE)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_multiple_basemodels_raises_validation_error(self, mock_collect):
        """Code with multiple BaseModel subclasses should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            ComponentSourceCode(code=MULTIPLE_BASEMODEL_CODE)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_no_atomic_workflow_raises_validation_error(self, mock_collect):
        """Code without @atomic_workflow decorator should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            ComponentSourceCode(code=NO_WORKFLOW_CODE)


class TestGeneratedComponentRulesCollection(unittest.TestCase):
    """Tests that new _validation_rules entries are collected."""

    def test_validation_rules_include_new_entries(self):
        """New _validation_rules should be collected by collect_all_rules()."""
        rules = ComponentSourceCode.collect_all_rules()

        self.assertGreaterEqual(len(rules), 6)

        self.assertTrue(
            any("fields from multiple unrelated business domains" in r for r in rules),
            "Should have High Cohesion rule about fields serving single artifact",
        )

        self.assertTrue(
            any("one rule" in r.lower() for r in rules),
            "Should have one-rule-per-entry meta-rule",
        )

        self.assertTrue(
            any("every word" in r.lower() for r in rules),
            "Should have conciseness rule",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
