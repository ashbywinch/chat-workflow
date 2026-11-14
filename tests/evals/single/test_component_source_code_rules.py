"""Eval tests for ComponentSourceCode LLM-judged validation rules.

Tests that High Cohesion and Complete Functionality are enforced by the judge
during model construction.

These tests call real LLM APIs and require config.json + API key.
"""

import unittest
from typing import override

from tests.conftest import timeout
from tests.evals.helpers import make_config
from tests.sample_code import (
    INCOMPLETE_CODE,
    LOW_COHESION_CODE,
    VALID_COMPONENT_CODE,
)
from workflows.workflow.component_source_code import ComponentSourceCode


class TestComponentSourceCodeLlmValidationEval(unittest.TestCase):
    """Eval tests for LLM-judged _validation_rules on ComponentSourceCode.

    These tests call real LLM APIs and verify that the validation rules
    catch structural and quality issues in generated component code.
    """

    @override
    def setUp(self):
        self.config = make_config()

    @timeout(120)
    def test_valid_code_passes_llm_validation(self):
        """Code that satisfies all rules should pass LLM validation."""
        component = ComponentSourceCode(code=VALID_COMPONENT_CODE)
        self.assertIsInstance(component, ComponentSourceCode)
        self.assertIn("class MinutesDraft", component.code)

    @timeout(120)
    def test_low_cohesion_code_fails_llm_validation(self):
        """Code mixing the artifact with cross-cutting concerns fails High Cohesion."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            ComponentSourceCode(code=LOW_COHESION_CODE)

        self.assertIn("cross-cutting", str(ctx.exception).lower())

    @timeout(120)
    def test_incomplete_code_fails_llm_validation(self):
        """Code with only type checks should fail Complete Functionality."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            ComponentSourceCode(code=INCOMPLETE_CODE)

        self.assertIn("domain business rule", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
