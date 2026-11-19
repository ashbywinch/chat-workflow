"""The transcript: PydanticAI's message list, projected for people and judges.

Pinned here: which parts become which entries, that thinking and tool traffic
survive the audit view, that the user view contains only what a person saw,
and that the stored form round-trips byte-for-byte.
"""

import unittest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from chat_workflow.transcript import dump, entries, load, render
from tests.conftest import make_valid_criteria


def _messages() -> list:
    return [
        ModelRequest(parts=[UserPromptPart("Grandma said so")]),
        ModelResponse(
            parts=[
                ThinkingPart("I should look this up"),
                ToolCallPart(tool_name="search", args={"query": "walter"}, tool_call_id="c1"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="search", content={"hit": "the war record"}, tool_call_id="c1"),
                RetryFeedbackPart(
                    content="the envelope is missing 'result'", retry=RetryInfo(cause="model_retry")
                ),
            ]
        ),
        ModelResponse(parts=[TextPart("Rejected answer")]),
        ModelResponse(parts=[TextPart('{"intent": "success", "result": {"ok": true}}')]),
    ]


class TestEntries(unittest.TestCase):
    def test_each_part_is_its_own_kind(self):
        kinds = [kind for kind, _ in entries(_messages())]

        self.assertEqual(
            kinds,
            ["user", "thinking", "tool-call", "tool-return", "validation", "assistant", "output"],
        )

    def test_a_tool_call_carries_its_name_and_arguments(self):
        call = next(text for kind, text in entries(_messages()) if kind == "tool-call")

        self.assertIn("search", call)
        self.assertIn("walter", call)

    def test_the_final_structured_answer_is_output_not_a_turn(self):
        kinds = [kind for kind, _ in entries(_messages())]

        self.assertEqual(kinds.count("output"), 1)
        self.assertEqual(kinds.count("assistant"), 1)


class TestRender(unittest.TestCase):
    def test_the_audit_view_keeps_thinking_tools_and_retries(self):
        text = render(_messages())

        self.assertIn("[thinking]\nI should look this up", text)
        self.assertIn("[tool-call]", text)
        self.assertIn("[tool-return]", text)
        self.assertIn("[validation]", text)
        self.assertIn("missing 'result'", text)

    def test_the_user_view_is_only_what_a_person_saw(self):
        text = render(_messages(), audience="user")

        self.assertIn("[user]\nGrandma said so", text)
        self.assertIn("[assistant]\nRejected answer", text)
        self.assertNotIn("thinking", text)
        self.assertNotIn("tool-call", text)
        self.assertNotIn("missing 'result'", text)
        self.assertNotIn("intent", text)

    def test_the_result_replaces_the_raw_answer(self):
        result = make_valid_criteria()

        audit = render(_messages(), result=result)
        audience = render(_messages(), audience="user", result=result)

        self.assertIn("[structured-result]", audit)
        self.assertIn("budget", audit)
        self.assertNotIn('"intent": "success"', audit)
        self.assertIn("[structured-result]", audience)
        self.assertNotIn("budget", audience)

    def test_a_failed_run_keeps_its_last_answer_as_evidence(self):
        text = render(_messages())

        self.assertIn("[output]", text)
        self.assertIn('{"intent": "success", "result": {"ok": true}}', text)


class TestStoredForm(unittest.TestCase):
    def test_the_message_list_round_trips_byte_for_byte(self):
        messages = _messages()

        stored = dump(messages)
        again = load(stored)

        self.assertEqual(dump(again), stored)

    def test_thinking_and_tool_parts_survive_the_round_trip(self):
        again = load(dump(_messages()))
        kinds = [kind for kind, _ in entries(again)]

        self.assertIn("thinking", kinds)
        self.assertIn("tool-call", kinds)
        self.assertIn("tool-return", kinds)
        self.assertIn("validation", kinds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
