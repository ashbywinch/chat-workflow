"""Eval tests for GeneratedComponent LLM-judged _validation_rules.

Tests that the new _validation_rules entries (High Cohesion, Complete
Functionality, Context Effectiveness, All playbooks on same artifact)
are enforced by the LLM validator during model construction.

These tests call real LLM APIs and require config.json + API key.
"""

import unittest

from tests.conftest import timeout
from tests.evals.helpers import make_config
from tests.sample_code import (
    INCOMPLETE_CODE,
    LOW_COHESION_CODE,
    MULTI_ARTIFACT_CODE,
    VALID_COMPONENT_CODE,
    VERBOSE_CODE,
)
from workflows.workflow.generated_component import GeneratedComponent


class TestGeneratedComponentLlmValidationEval(unittest.TestCase):
    """Eval tests for LLM-judged _validation_rules on GeneratedComponent.

    These tests call real LLM APIs and verify that the validation rules
    catch structural and quality issues in generated component code.
    """

    def setUp(self):
        self.config = make_config()

    @timeout(120)
    def test_valid_code_passes_llm_validation(self):
        """Code that satisfies all rules should pass LLM validation."""
        component = GeneratedComponent(code=VALID_COMPONENT_CODE)
        self.assertIsInstance(component, GeneratedComponent)
        self.assertIn("class MeetingMinutes", component.code)

    @timeout(120)
    def test_low_cohesion_code_fails_llm_validation(self):
        """Code with fields from multiple domains should fail High Cohesion."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            GeneratedComponent(code=LOW_COHESION_CODE)

    @timeout(120)
    def test_incomplete_code_fails_llm_validation(self):
        """Code with no validation or lifecycle should fail Complete Functionality."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            GeneratedComponent(code=INCOMPLETE_CODE)

    @timeout(120)
    def test_verbose_code_fails_llm_validation(self):
        """Code with verbose, redundant docstrings should fail Context Effectiveness."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            GeneratedComponent(code=VERBOSE_CODE)

    @timeout(120)
    def test_multi_artifact_code_fails_llm_validation(self):
        """Code with @atomic_workflow methods for different domains should fail."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError):
            GeneratedComponent(code=MULTI_ARTIFACT_CODE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
