#!/usr/bin/env python3
"""Tests for InstructorRetryException handling and LLM self-correction via model_validator."""

import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.exceptions import MaxRetriesExceededError
from chat_workflow.models import AgentIntent, AgentResponse
from tests.conftest import make_atomic_workflow_config, make_valid_criteria
from workflows.evaluation_criteria import EvaluationCriteria


class TestAgentResponseModelValidator(unittest.TestCase):
    """Tests for the model_validator(mode='before') on AgentResponse."""

    def test_bare_list_raises_validation_error(self):
        """Bare list input raises ValidationError with actionable message."""
        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[EvaluationCriteria].model_validate(["item1", "item2"])
        error_msg = str(ctx.exception)
        self.assertIn("bare list", error_msg.lower())

    def test_inner_fields_without_wrapper_raises_validation_error(self):
        """Dict with inner/domain fields but no intent/result raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[EvaluationCriteria].model_validate(
                {"consumer": "test consumer", "format": "json", "source": "test"}
            )
        error_msg = str(ctx.exception)
        self.assertIn("AgentResponse", error_msg)
        self.assertIn("consumer", error_msg)

    def test_valid_input_passes_through_validator(self):
        """Valid AgentResponse input passes through the before-validator successfully."""
        response = AgentResponse[EvaluationCriteria].model_validate(
            {"intent": "success", "result": make_valid_criteria().model_dump()}
        )
        self.assertEqual(response.intent, AgentIntent.SUCCESS)
        self.assertIsNotNone(response.result)

    def test_after_validator_still_works(self):
        """The existing mode='after' validator still enforces intent consistency."""
        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[EvaluationCriteria].model_validate(
                {"intent": "success", "result": None}
            )
        self.assertIn("field named 'result'", str(ctx.exception))


class TestCallLlmInstructorRetryHandling(unittest.TestCase):
    """Tests for InstructorRetryException -> MaxRetriesExceededError in _call_llm."""

    @patch("chat_workflow.llm_interaction.get_client")
    def test_instructor_retry_converted_to_max_retries(self, mock_get_client):
        """InstructorRetryException in _call_llm raises MaxRetriesExceededError."""
        from instructor.core.exceptions import InstructorRetryException

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = InstructorRetryException(
            "test error", n_attempts=3, total_usage=0
        )
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(max_retries=3))

        with self.assertRaises(MaxRetriesExceededError) as ctx:
            orchestrator._call_llm()

        self.assertIn("3", str(ctx.exception))
        self.assertIn("Maximum retries", str(ctx.exception))

    @patch("chat_workflow.llm_interaction.get_client")
    def test_generic_exceptions_still_propagate(self, mock_get_client):
        """Generic exceptions still propagate through _call_llm unchanged."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Unexpected error")
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        with self.assertRaises(RuntimeError):
            orchestrator._call_llm()


if __name__ == "__main__":
    unittest.main(verbosity=2)
