"""Tests for ProcessAnalysis.generate_from_chat atomic workflow."""

import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.models import AgentIntent, AgentResponse
from workflows.workflow.models import ProcessAnalysis


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class TestAnalyzeProcess(unittest.TestCase):
    def _make_session(self) -> Session:
        return Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def test_has_workflow_attribute(self):
        """generate_from_chat should be discoverable as a workflow function."""
        self.assertTrue(
            getattr(ProcessAnalysis.generate_from_chat, "_is_workflow", False),
            "generate_from_chat should have _is_workflow=True",
        )

    def test_requires_session(self):
        """Calling without session should raise TypeError."""
        with self.assertRaises(TypeError) as ctx:
            ProcessAnalysis.generate_from_chat(process_description="test process")
        self.assertIn("session", str(ctx.exception))

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_returns_process_analysis(self, mock_call_llm):
        """With mocked LLM returning ProcessAnalysis, method should return it."""
        expected = ProcessAnalysis(
            phases=["Intake", "Processing", "Completion"],
            activities=["Receive request", "Validate data", "Process payment"],
            orchestrating_component="Order Management",
            participants=["Customer", "Order System", "Payment Gateway"],
        )
        mock_call_llm.return_value = AgentResponse[ProcessAnalysis](
            intent=AgentIntent.SUCCESS,
            result=expected,
        )
        session = self._make_session()
        result = ProcessAnalysis.generate_from_chat(
            process_description="Customer order processing",
            session=session,
        )
        self.assertIsInstance(result, ProcessAnalysis)
        self.assertEqual(result.orchestrating_component, "Order Management")
        self.assertEqual(len(result.phases), 3)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_passes_process_description_to_llm(self, mock_call_llm):
        """The process_description should be passed through to the LLM invocation."""
        mock_call_llm.return_value = AgentResponse[ProcessAnalysis](
            intent=AgentIntent.SUCCESS,
            result=ProcessAnalysis(
                phases=["Test"],
                activities=["Test"],
                orchestrating_component="Test",
                participants=["Test"],
            ),
        )
        session = self._make_session()
        ProcessAnalysis.generate_from_chat(
            process_description="Customer onboarding process for new users",
            session=session,
        )
        mock_call_llm.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
