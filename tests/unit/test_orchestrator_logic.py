#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, patch

from evaluation_criteria.models import EvaluationCriteria, Criterion
from prompt_core import (
    ConversationAction,
    ConversationResult,
    StructuredConversationOrchestrator,
)
from prompt_core.exceptions import (
    ConversationFailedError,
    TurnLimitExceededError,
    InvalidResponseError,
)


class TestStructuredConversationOrchestrator(unittest.TestCase):
    def setUp(self):
        self.valid_criteria = EvaluationCriteria(
            context="test context",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )

    def test_orchestrator_initialization(self):
        from prompt_core.config import config

        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=5,
            initial_messages=[{"role": "user", "content": "Initial context: test"}],
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        self.assertEqual(orchestrator.turn_count, 0)
        self.assertEqual(orchestrator.max_turns, 5)
        self.assertEqual(orchestrator.model, config.model)
        self.assertEqual(len(orchestrator.messages), 2)
        self.assertEqual(orchestrator.messages[0]["role"], "system")
        self.assertEqual(orchestrator.messages[1]["role"], "user")
        self.assertIn("Initial context", orchestrator.messages[1]["content"])

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_process_turn_success(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=10,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        action = ConversationAction[EvaluationCriteria](
            action="success", result=self.valid_criteria
        )
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("Let's create criteria")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, self.valid_criteria)
        self.assertEqual(orchestrator.turn_count, 1)
        self.assertEqual(len(orchestrator.messages), 2)

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_process_turn_continue(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=10,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        action = ConversationAction[EvaluationCriteria](
            action="continue", message="What's your budget range?"
        )
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("Hello")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What's your budget range?")
        self.assertIsNone(result.result)
        self.assertEqual(orchestrator.turn_count, 1)
        self.assertEqual(len(orchestrator.messages), 3)
        self.assertEqual(orchestrator.messages[2]["role"], "assistant")

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_process_turn_failure_raises_exception(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=10,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        action = ConversationAction[EvaluationCriteria](
            action="failure", message="I don't have enough information to help"
        )
        mock_call_llm.return_value = action

        with self.assertRaises(ConversationFailedError) as context:
            orchestrator.process_turn("Something vague")

        self.assertIn(
            "I don't have enough information to help",
            str(context.exception),
        )
        self.assertEqual(orchestrator.turn_count, 1)

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_process_turn_empty_input(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=10,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        action = ConversationAction[EvaluationCriteria](
            action="continue", message="First question"
        )
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "First question")
        self.assertEqual(len(orchestrator.messages), 2)

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_multi_turn_conversation_sequence(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=5,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        responses = [
            ConversationAction[EvaluationCriteria](
                action="continue", message="Question 1"
            ),
            ConversationAction[EvaluationCriteria](
                action="continue", message="Question 2"
            ),
            ConversationAction[EvaluationCriteria](
                action="success", result=self.valid_criteria
            ),
        ]
        mock_call_llm.side_effect = responses

        result1 = orchestrator.process_turn("Hello")
        self.assertFalse(result1.is_complete)
        self.assertEqual(result1.message, "Question 1")
        self.assertEqual(orchestrator.turn_count, 1)

        result2 = orchestrator.process_turn("Answer 1")
        self.assertFalse(result2.is_complete)
        self.assertEqual(result2.message, "Question 2")
        self.assertEqual(orchestrator.turn_count, 2)

        result3 = orchestrator.process_turn("Answer 2")
        self.assertTrue(result3.is_complete)
        self.assertEqual(result3.result, self.valid_criteria)
        self.assertEqual(orchestrator.turn_count, 3)

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_orchestrator_raises_on_max_turns(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=2,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        mock_call_llm.return_value = ConversationAction[EvaluationCriteria](
            action="continue", message="Another question"
        )

        result1 = orchestrator.process_turn("A1")
        self.assertFalse(result1.is_complete)
        self.assertEqual(orchestrator.turn_count, 1)

        result2 = orchestrator.process_turn("A2")
        self.assertFalse(result2.is_complete)
        self.assertEqual(orchestrator.turn_count, 2)

        with self.assertRaises(TurnLimitExceededError) as context:
            orchestrator.process_turn("A3")

        self.assertIn("Maximum conversation turns (2) reached", str(context.exception))

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_process_turn_propagates_llm_exceptions(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=10,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        mock_call_llm.side_effect = ValueError("Validation failed after retries")

        with self.assertRaises(ValueError):
            orchestrator.process_turn("test")

        self.assertEqual(orchestrator.turn_count, 1)

    @patch.object(StructuredConversationOrchestrator, "_call_llm")
    def test_invalid_action_raises_exception(self, mock_call_llm):
        orchestrator = StructuredConversationOrchestrator(
            system_prompt="Test prompt",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=10,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
        )

        mock_action = Mock()
        mock_action.action = "invalid_action"
        mock_action.message = "test"
        mock_action.result = None
        mock_call_llm.return_value = mock_action

        with self.assertRaises(InvalidResponseError) as context:
            orchestrator.process_turn("test")

        self.assertIn("Invalid action received: invalid_action", str(context.exception))


class TestWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        self.valid_criteria = EvaluationCriteria(
            context="test context",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )

    @patch(
        "prompt_core.conversation_runtime.StructuredConversationOrchestrator._call_llm"
    )
    def test_leaf_accepts_tools_parameter(self, mock_call_llm):
        from prompt_core import ConversationTools, ConversationFlowState
        from evaluation_criteria.flows import generate_criteria

        mock_call_llm.return_value = ConversationAction[EvaluationCriteria](
            action="success", result=self.valid_criteria
        )

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="test response")

        state = ConversationFlowState()
        tools = ConversationTools(io=mock_io, state=state)

        result = generate_criteria(context="test", max_turns=5, tools=tools)

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(len(result.criteria), 2)

    @patch(
        "prompt_core.conversation_runtime.StructuredConversationOrchestrator._call_llm"
    )
    def test_leaf_accepts_io_parameter(self, mock_call_llm):
        from evaluation_criteria.flows import generate_criteria

        mock_call_llm.return_value = ConversationAction[EvaluationCriteria](
            action="success", result=self.valid_criteria
        )

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="test response")

        result = generate_criteria(context="test", max_turns=5, io=mock_io)

        self.assertIsInstance(result, EvaluationCriteria)

    @patch(
        "prompt_core.conversation_runtime.StructuredConversationOrchestrator._call_llm"
    )
    def test_workflow_passes_tools_to_leaf(self, mock_call_llm):
        from evaluation_criteria.flows import run_reviewed_criteria_conversation
        from prompt_core import ConversationFlowState

        mock_call_llm.return_value = ConversationAction[EvaluationCriteria](
            action="success", result=self.valid_criteria
        )

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="looks good")

        state = ConversationFlowState()

        result = run_reviewed_criteria_conversation(
            context="test context",
            max_turns=5,
            io=mock_io,
            state=state,
        )

        self.assertIsInstance(result, EvaluationCriteria)


if __name__ == "__main__":
    unittest.main(verbosity=2)
