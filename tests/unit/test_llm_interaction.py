#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import tests.conftest
from chat_workflow import (
    AgentIntent,
    AgentResponse,
    AtomicWorkflow,
    AtomicWorkflowFailedError,
    TurnLimitExceededError,
)
from workflows.evaluation_criteria import EvaluationCriteria


class TestLLMInteraction(unittest.TestCase):
    def _create_orchestrator(self, max_turns=10):
        return AtomicWorkflow(
            config=tests.conftest.make_atomic_workflow_config(max_turns=max_turns)
        )

    @patch("chat_workflow.llm_interaction.get_client")
    def test_call_llm_with_validation_error(self, mock_get_client):
        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [ValueError("Validation failed after 3 retries")]
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator()

        with self.assertRaises(ValueError) as context:
            orchestrator._call_llm()

        self.assertIn("Validation failed", str(context.exception))

    @patch("chat_workflow.llm_interaction.get_client")
    def test_call_llm_with_api_error(self, mock_get_client):
        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [Exception("API error: Invalid API key")]
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator()

        with self.assertRaises(Exception) as context:
            orchestrator._call_llm()

        self.assertIn("API error", str(context.exception))

    @patch("chat_workflow.llm_interaction.get_client")
    def test_call_llm_passes_correct_parameters(self, mock_get_client):
        mock_client = tests.conftest.MockInstructorClient()
        expected_action = AgentResponse[EvaluationCriteria](intent=AgentIntent.CONTINUE, message="Test")
        mock_client.responses = [expected_action]
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator()

        orchestrator._call_llm()

        self.assertEqual(mock_client.call_count, 1)
        model, messages, response_model, max_retries = mock_client.last_call_args

        self.assertEqual(model, "default-model")
        self.assertEqual(messages, orchestrator.messages)
        self.assertEqual(response_model, AgentResponse[EvaluationCriteria])
        self.assertEqual(max_retries, 3)
        self.assertEqual(
            mock_client.last_call_kwargs.get("timeout"),
            30,
        )


    @patch("chat_workflow.llm_interaction.get_client")
    def test_conversation_success_completion(self, mock_get_client):
        success_action = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.SUCCESS, result=tests.conftest.make_valid_criteria()
        )

        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [success_action]
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator()

        result = orchestrator.process_turn("Let's create criteria for test")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, tests.conftest.make_valid_criteria())
        self.assertIn("success", result.message.lower())

        self.assertEqual(mock_client.call_count, 1)
        _, _, response_model, max_retries = mock_client.last_call_args
        self.assertEqual(response_model, AgentResponse[EvaluationCriteria])
        self.assertEqual(max_retries, 3)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_conversation_turn_limit_enforcement(self, mock_get_client):
        continue_action = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.CONTINUE, message="Tell me more"
        )
        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [continue_action] * 20
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator(max_turns=10)

        for i in range(10):
            result = orchestrator.process_turn(f"User input {i}")
            self.assertFalse(result.is_complete)
            self.assertEqual(result.message, "Tell me more")

        with self.assertRaises(TurnLimitExceededError) as context:
            orchestrator.process_turn("One more turn")

        self.assertIn("10", str(context.exception))
        self.assertEqual(mock_client.call_count, 10)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_conversation_with_custom_max_turns(self, mock_get_client):
        continue_action = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.CONTINUE,
            message="Continue please",
        )
        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [continue_action] * 5
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator(max_turns=3)

        for i in range(3):
            result = orchestrator.process_turn(f"Input {i}")
            self.assertFalse(result.is_complete)

        with self.assertRaises(TurnLimitExceededError) as context:
            orchestrator.process_turn("Fourth turn")

        self.assertIn("3", str(context.exception))
        self.assertEqual(mock_client.call_count, 3)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_conversation_failure_action_termination(self, mock_get_client):
        failure_action = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.FAILURE, message="Cannot generate criteria with given information"
        )

        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [failure_action]
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator()

        with self.assertRaises(AtomicWorkflowFailedError) as context:
            orchestrator.process_turn("Some user input")

        self.assertIn("Cannot generate criteria", str(context.exception))
        self.assertEqual(mock_client.call_count, 1)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_validation_error_propagation(self, mock_get_client):
        mock_client = tests.conftest.MockInstructorClient()
        mock_client.responses = [ValueError("Validation failed: Invalid JSON structure")]
        mock_get_client.return_value = mock_client

        orchestrator = self._create_orchestrator()

        with self.assertRaises(ValueError) as context:
            orchestrator.process_turn("Test input")

        self.assertIn("Validation failed", str(context.exception))
        self.assertEqual(mock_client.call_count, 1)


class TestGetClient(unittest.TestCase):
    @patch("chat_workflow.llm_interaction.instructor.from_litellm")
    def test_uses_custom_api_key_env_and_api_base(self, mock_from_litellm):
        import os

        import chat_workflow.llm_interaction as li

        mock_client = object()
        mock_from_litellm.return_value = mock_client
        saved_base = li.litellm.api_base
        try:
            with patch.dict(os.environ, {"OPENCODE_GO_EVALS_API_KEY": "test-key"}, clear=False):
                client = li.get_client(
                    "openai",
                    api_base="https://opencode.ai/zen/go/v1",
                    api_key_env="OPENCODE_GO_EVALS_API_KEY",
                )
                # Key exposed to litellm under the provider's canonical env name
                self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test-key")
                self.assertEqual(li.litellm.api_base, "https://opencode.ai/zen/go/v1")
            self.assertIs(client, mock_client)
        finally:
            li.litellm.api_base = saved_base

    @patch("chat_workflow.llm_interaction.instructor.from_litellm")
    def test_uses_provider_default_key_env_when_not_overridden(self, mock_from_litellm):
        import os

        import chat_workflow.llm_interaction as li

        mock_from_litellm.return_value = object()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "default-key"}, clear=False):
            li.get_client("openai")
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "default-key")

    def test_missing_key_raises_api_key_error(self):
        import os

        from chat_workflow import APIKeyError
        from chat_workflow.llm_interaction import get_client

        with patch.dict(os.environ, {}, clear=True), self.assertRaises(APIKeyError):
            get_client("openai", api_key_env="OPENCODE_GO_EVALS_API_KEY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
