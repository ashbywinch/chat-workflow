#!/usr/bin/env python3
"""The call trace is the judge's view: input, conversation and output per step.

A single session can hold several steps. The trace must show which turns
belonged to which call, and what went in and out, instead of one flat
conversation with a structured result at the end.
"""

import unittest
from typing import override

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

from chat_workflow.session import Session, UserIO
from chat_workflow.session_log import SessionLog, TranscriptSegment
from chat_workflow.transcript import render_call_trace


class _IO(UserIO):
    @override
    def echo(self, message: str) -> None:
        pass

    @override
    def prompt(self, label: str) -> str:
        return ""


def _session(log: SessionLog) -> Session:
    return Session(io=_IO(), state=log)


class TestCallTrace(unittest.TestCase):
    def test_each_step_is_a_frame_with_its_own_input_conversation_and_output(self):
        log = SessionLog()
        log.messages = [
            ModelRequest.user_text_prompt("Team events are chaos"),
            ModelResponse(parts=[TextPart("What do you make?")]),
            ModelResponse(parts=[TextPart("Here's what I've got so far.")]),
            ModelRequest.user_text_prompt("Yeah, that's it"),
            ModelRequest.user_text_prompt("Start the conversation"),
            ModelResponse(parts=[TextPart("I have what I need.")]),
        ]
        log.segments = [
            TranscriptSegment(
                0,
                2,
                step="_gather_notes",
                inputs={"process_description": ""},
                output={"notes": "Team events are chaos"},
            ),
            TranscriptSegment(
                4,
                6,
                chat=False,
                step="sign_off",
                inputs={},
                output={"text": "I have what I need."},
            ),
        ]

        trace = render_call_trace(
            _session(log),
            result={"stages": ["Find a time"]},
            given={"context": "team events"},
            title="generate_from_chat",
        )

        self.assertIn("Call trace: generate_from_chat", trace)
        self.assertIn("Input:\n  {\n    \"context\": \"team events\"", trace)

        self.assertIn("→ _gather_notes (chat)", trace)
        self.assertIn('"process_description": ""', trace)
        self.assertIn("[user]\n    Team events are chaos", trace)
        self.assertIn('"notes": "Team events are chaos"', trace)
        self.assertIn("← _gather_notes", trace)

        # The composite's own proposal and the user's reply sit at the caller
        # level, not inside either step.
        self.assertIn("[caller]\n  [assistant]\n  Here's what I've got so far.", trace)
        self.assertIn("  [user]\n  Yeah, that's it", trace)

        self.assertIn("→ sign_off (chat-off)", trace)
        self.assertIn("Conversation:\n    (chat-off; no user turns)", trace)
        self.assertIn('"text": "I have what I need."', trace)

        self.assertIn("Result:\n  {\n    \"stages\": [", trace)

    def test_a_chat_step_with_no_turns_says_so(self):
        log = SessionLog()
        log.messages = []
        log.segments = [TranscriptSegment(0, 0, step="empty", inputs={})]

        trace = render_call_trace(_session(log))

        self.assertIn("→ empty (chat)", trace)
        self.assertIn("Conversation:\n    (no turns)", trace)
        self.assertIn("Output:\n    (none)", trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
