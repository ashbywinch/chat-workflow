#!/usr/bin/env python3
import unittest
from pathlib import Path

from chat_workflow import (
    AgentIntent,
    AgentResponse,
    AtomicWorkflow,
    AtomicWorkflowConfig,
    AtomicWorkflowFailedError,
    Config,
    Session,
    SessionLog,
    TurnLimitExceededError,
    TurnResult,
)
from tests.conftest import timeout
from workflows.evaluation_criteria import EvaluationCriteria

_CONFIG = Config(Path(__file__).parent.parent.parent / "config.json")


class MockIO:
    def __init__(self, responses):
        self.responses = list(responses)
        self.outputs = []

    def echo(self, message: str) -> None:
        self.outputs.append(message)

    def prompt(self, label: str) -> str:
        if self.responses:
            return self.responses.pop(0)
        return ""


class TestRealAPI(unittest.TestCase):
    @timeout(10)
    def test_call_llm_returns_valid_action(self):
        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are a helpful assistant that creates evaluation criteria. "
                    "When returning intent='success' with criteria, you MUST include "
                    "a criterion named 'budget' (lowercase). "
                    "Use intent='continue' to ask questions, intent='success' to return criteria, "
                    "intent='failure' if unable to help."
                ),
                response_model=AgentResponse[EvaluationCriteria],
                max_turns=5,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": "Create criteria for choosing a birthday gift. My budget is $50.",
                    }
                ],
                on_continue=lambda action: TurnResult[EvaluationCriteria].continuing(action.message),
                on_success=lambda action: TurnResult[EvaluationCriteria].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message),
            )
        )

        action = orchestrator._call_llm()

        self.assertIn(action.intent, [AgentIntent.CONTINUE, AgentIntent.SUCCESS, AgentIntent.FAILURE])

        if action.intent in [AgentIntent.CONTINUE, AgentIntent.FAILURE]:
            self.assertIsNotNone(action.message)
            self.assertTrue(len(action.message) > 0)

        if action.intent == AgentIntent.SUCCESS:
            self.assertIsNotNone(action.result)
            self.assertGreaterEqual(len(action.result.criteria), 2)
            has_budget = any(c.name.lower() == "budget" for c in action.result.criteria)
            self.assertTrue(
                has_budget,
                f"Criteria must include 'budget'. Found: {[c.name for c in action.result.criteria]}",
            )

        action = orchestrator._call_llm()

        self.assertIn(action.intent, [AgentIntent.CONTINUE, AgentIntent.SUCCESS, AgentIntent.FAILURE])

        if action.intent in [AgentIntent.CONTINUE, AgentIntent.FAILURE]:
            self.assertIsNotNone(action.message)
            self.assertTrue(len(action.message) > 0)

        if action.intent == AgentIntent.SUCCESS:
            self.assertIsNotNone(action.result)
            self.assertGreaterEqual(len(action.result.criteria), 2)
            has_budget = any(c.name.lower() == "budget" for c in action.result.criteria)
            self.assertTrue(
                has_budget,
                f"Criteria must include 'budget'. Found: {[c.name for c in action.result.criteria]}",
            )

    @timeout(10)
    def test_multi_turn_conversation_with_real_llm(self):
        mock_io = MockIO(
            [
                "Around $50 for the budget",
                "For a 7-year-old who likes science",
                "Safety is important",
                "That's all, please finalize with budget criterion",
            ]
        )

        criteria = EvaluationCriteria.generate_from_chat(
            context="choosing a birthday gift",
            max_turns=6,
            session=Session(io=mock_io, state=SessionLog(), config=_CONFIG),
        )

        self.assertIsInstance(criteria, EvaluationCriteria)
        self.assertGreaterEqual(len(criteria.criteria), 2)
        has_budget = any(c.name.lower() == "budget" for c in criteria.criteria)
        self.assertTrue(
            has_budget,
            f"Criteria must include 'budget'. Found: {[c.name for c in criteria.criteria]}",
        )

    @timeout(10)
    def test_single_turn_with_real_llm(self):
        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are a helpful assistant for creating evaluation criteria. "
                    "When returning intent='success' with criteria, you MUST include "
                    "a criterion named 'budget' (lowercase). "
                    "Use intent='continue' to ask questions, intent='success' to return criteria, "
                    "intent='failure' if unable to help."
                ),
                response_model=AgentResponse[EvaluationCriteria],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": "I want to evaluate coffee makers. Budget is $200.",
                    }
                ],
                on_continue=lambda action: TurnResult[EvaluationCriteria].continuing(action.message),
                on_success=lambda action: TurnResult[EvaluationCriteria].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message),
            )
        )

        result = orchestrator.process_turn("")

        self.assertIsInstance(result, TurnResult)
        self.assertIsNotNone(result.message)
        self.assertTrue(len(result.message) > 0)

        if result.result:
            self.assertIsInstance(result.result, EvaluationCriteria)
            self.assertGreaterEqual(len(result.result.criteria), 2)
            self.assertTrue(
                any(c.name.lower() == "budget" for c in result.result.criteria),
                f"Criteria must include 'budget'. Found: {[c.name for c in result.result.criteria]}",
            )

    @timeout(10)
    def test_conversation_flow_with_real_llm(self):
        mock_io = MockIO(
            [
                "My budget is around $50",
                "They like building toys and science kits",
                "Safety is important for a 7-year-old",
                "Educational value would be good",
                "Please finalize the criteria now with budget included",
            ]
        )

        criteria = EvaluationCriteria.generate_from_chat(
            context="choosing a birthday gift for a 7-year-old",
            max_turns=10,
            session=Session(io=mock_io, state=SessionLog(), config=_CONFIG),
        )

        self.assertIsInstance(criteria, EvaluationCriteria)
        self.assertGreaterEqual(len(criteria.criteria), 2)

        budget_found = any(c.name.lower() == "budget" for c in criteria.criteria)
        self.assertTrue(
            budget_found,
            f"Criteria missing 'budget'. Found: {[c.name for c in criteria.criteria]}",
        )

    @timeout(10)
    def test_uncooperative_user_max_turns(self):
        mock_io = MockIO(
            [
                "I'm not sure",
                "Maybe something good",
                "Whatever you think",
            ]
        )

        with self.assertRaises((TurnLimitExceededError, AtomicWorkflowFailedError)):
            EvaluationCriteria.generate_from_chat(
                context="choosing a laptop for programming",
                max_turns=3,
                session=Session(io=mock_io, state=SessionLog(), config=_CONFIG),
            )

    @timeout(10)
    def test_conversation_action_format(self):
        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are a helpful assistant for creating evaluation criteria. "
                    "When returning intent='success' with criteria, you MUST include "
                    "a criterion named 'budget' (lowercase). "
                    "Use intent='continue' to ask questions, intent='success' to return criteria, "
                    "intent='failure' if unable to help."
                ),
                response_model=AgentResponse[EvaluationCriteria],
                max_turns=5,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": "I want to create criteria for choosing a laptop. My budget is $1000.",
                    }
                ],
                on_continue=lambda action: TurnResult[EvaluationCriteria].continuing(action.message),
                on_success=lambda action: TurnResult[EvaluationCriteria].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message),
            )
        )

        result = orchestrator.process_turn("Please include budget as a criterion")
        self.assertIsInstance(result, TurnResult)
        self.assertIsNotNone(result.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
