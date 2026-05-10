#!/usr/bin/env python3
import unittest
from io import StringIO
from pathlib import Path

from chat_workflow import (
    ConversationAction,
    ConversationFlowState,
    ConversationTools,
    StreamingDebug,
)
from chat_workflow.config import Config
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

        action = ConversationAction[EvaluationCriteria](action="continue", message="What is your budget?")
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
        """Verify debug output captures LLM interaction when using @chat decorator."""
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        mock_io = MockIO(
            [
                "Performance, battery life, and portability",
                "I need it for software development and travel",
                "That's all, please finalize",
            ]
        )

        try:
            criteria = EvaluationCriteria.generate_from_chat(
                context="choosing a laptop",
                max_turns=6,
                tools=ConversationTools(io=mock_io, state=ConversationFlowState(), config=_CONFIG),
                debug=debug,
            )

            output = debug_output.getvalue()
            self.assertIn("LLM REQUEST", output)
            self.assertIn("LLM RESPONSE", output)
            self.assertIsInstance(criteria, EvaluationCriteria)
        except Exception:
            output = debug_output.getvalue()
            self.assertIn("LLM REQUEST", output)
            self.assertIn("LLM RESPONSE", output)
            raise

    @timeout(10)
    def test_chat_decorator_with_debug_parameter(self):
        debug_output = StringIO()
        debug = StreamingDebug(file=debug_output, include_timestamps=False)

        mock_io = MockIO(
            [
                "Around $50 budget",
                "For a 7-year-old who likes science",
                "Safety and educational value",
                "That's all, please finalize",
            ]
        )

        criteria = EvaluationCriteria.generate_from_chat(
            context="choosing a birthday gift",
            max_turns=6,
            tools=ConversationTools(io=mock_io, state=ConversationFlowState(), config=_CONFIG),
            debug=debug,
        )

        output = debug_output.getvalue()
        self.assertIn("LLM REQUEST", output)
        self.assertIn("LLM RESPONSE", output)

        self.assertIsInstance(criteria, EvaluationCriteria)


if __name__ == "__main__":
    unittest.main(verbosity=2)
