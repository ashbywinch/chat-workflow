"""Unit tests for StreamingDebug capture formatting (no API calls)."""

import unittest
from io import StringIO

from chat_workflow import AgentIntent, AgentResponse, StreamingDebug
from tests.conftest import timeout
from workflows.evaluation_criteria import EvaluationCriteria


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

        action = AgentResponse[EvaluationCriteria](intent=AgentIntent.CONTINUE, message="What is your budget?")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
