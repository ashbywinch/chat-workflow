#!/usr/bin/env python3
import unittest
from io import StringIO

from evaluation_criteria.models import EvaluationCriteria
from evaluation_criteria.flows import generate_criteria
from prompt_core import (
    ConversationAction,
    StructuredConversationOrchestrator,
    ConversationResult,
    StreamingDebug,
)
from prompt_core.exceptions import ConversationFailedError
from tests.conftest import timeout


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


class TestDebugStreaming(unittest.TestCase):
    @timeout(10)
    def test_streaming_debug_captures_request(self):
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        debug.on_request(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            model="test-model",
        )

        output = debug_output.getvalue()
        self.assertIn("LLM REQUEST", output)
        self.assertIn("test-model", output)
        self.assertIn("system: You are helpful", output)
        self.assertIn("user: Hello", output)

    @timeout(10)
    def test_streaming_debug_captures_response(self):
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        action = ConversationAction[EvaluationCriteria](
            action="continue", message="What is your budget?"
        )
        debug.on_response(action, duration_ms=123.45)

        output = debug_output.getvalue()
        self.assertIn("LLM RESPONSE", output)
        self.assertIn("123ms", output)
        self.assertIn("continue", output)

    @timeout(10)
    def test_streaming_debug_captures_error(self):
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        debug.on_error(ValueError("Something went wrong"))

        output = debug_output.getvalue()
        self.assertIn("ERROR", output)
        self.assertIn("ValueError", output)
        self.assertIn("Something went wrong", output)

    @timeout(10)
    def test_orchestrator_with_debug(self):
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        orchestrator = StructuredConversationOrchestrator(
            system_prompt="You are a helpful assistant.",
            response_model=ConversationAction[EvaluationCriteria],
            max_turns=3,
            initial_messages=[
                {
                    "role": "user",
                    "content": "Create criteria for choosing a laptop. Budget $1000.",
                }
            ],
            on_continue=lambda action: ConversationResult[
                EvaluationCriteria
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[EvaluationCriteria].success(
                action.result
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
            debug=debug,
        )

        try:
            orchestrator._call_llm()

            output = debug_output.getvalue()
            self.assertIn("LLM REQUEST", output)
            self.assertIn("LLM RESPONSE", output)
        except Exception:
            output = debug_output.getvalue()
            if "LLM REQUEST" in output:
                self.assertIn("ERROR", output)
            raise

    @timeout(10)
    def test_chat_decorator_with_debug_parameter(self):
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        mock_io = MockIO(
            [
                "Around $50 for the budget",
                "For a 7-year-old who likes science",
                "That's all, please finalize with budget criterion",
            ]
        )

        criteria = generate_criteria(
            context="choosing a birthday gift",
            max_turns=6,
            io=mock_io,
            debug=debug,
        )

        output = debug_output.getvalue()
        self.assertIn("LLM REQUEST", output)
        self.assertIn("LLM RESPONSE", output)

        self.assertIsInstance(criteria, EvaluationCriteria)


if __name__ == "__main__":
    unittest.main(verbosity=2)
