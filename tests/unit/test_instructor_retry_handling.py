#!/usr/bin/env python3
"""Tests for InstructorRetryException handling and LLM self-correction via model_validator."""

import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from chat_workflow.atomic_workflow_step import AtomicWorkflowStep
from chat_workflow.exceptions import MaxRetriesExceededError
from chat_workflow.models import AgentIntent, AgentResponse
from tests.conftest import make_atomic_workflow_step_config, make_valid_criteria
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

        orchestrator = AtomicWorkflowStep(config=make_atomic_workflow_step_config(max_retries=3))

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

        orchestrator = AtomicWorkflowStep(config=make_atomic_workflow_step_config())

        with self.assertRaises(RuntimeError):
            orchestrator._call_llm()

    @patch("chat_workflow.llm_interaction.get_client")
    def test_validation_failure_is_fed_back_before_retry(self, mock_get_client):
        """The rejected response and the reason are both appended for the retry."""
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow import ValidationError

        rejected_content = '{"intent": "success", "result": {"code": "nested if"}}'
        rejected = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            last_completion=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=rejected_content)
                    )
                ]
            ),
            failed_attempts=[FailedAttempt(1, ValidationError("meeting_date needs min_length"))],
        )
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            rejected,
            AgentResponse[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                result=make_valid_criteria(),
            ),
        ]
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, validation_corrections=1)
        )

        result = orchestrator._call_llm()

        self.assertEqual(result.intent, AgentIntent.SUCCESS)
        self.assertEqual(orchestrator.messages[-2]["role"], "assistant")
        self.assertEqual(orchestrator.messages[-2]["content"], rejected_content)
        correction = orchestrator.messages[-1]["content"]
        self.assertIn("meeting_date needs min_length", correction)
        self.assertIn("[framework]", correction)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_correction_budget_is_bounded(self, mock_get_client):
        """Exhausting the correction budget raises MaxRetriesExceededError."""
        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow import ValidationError

        def rejected():
            return InstructorRetryException(
                "rejected",
                n_attempts=1,
                total_usage=0,
                failed_attempts=[FailedAttempt(1, ValidationError("still wrong"))],
            )

        def always_reject(**_):
            raise rejected()

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = always_reject
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, validation_corrections=1)
        )

        with self.assertRaises(MaxRetriesExceededError) as ctx:
            orchestrator._call_llm()

        self.assertIn("still wrong", str(ctx.exception))
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_invalid_escape_reason_has_remediation(self):
        from json import JSONDecodeError

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow.atomic_workflow_step import _validation_failure

        exc = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            failed_attempts=[
                FailedAttempt(
                    1,
                    JSONDecodeError(
                        "Invalid JSON: invalid escape at line 5",
                        '{"x": "\\ "}',
                        5,
                    ),
                )
            ],
        )

        failure = _validation_failure(exc)
        assert failure is not None
        reason = failure.message()

        self.assertIn("Use `\\n`", reason)
        self.assertIn("backslash", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
