"""The transcript refuses a message list a provider would mishandle."""

import unittest

from chat_workflow.transcript import Transcript, TranscriptError


class TestValidShape(unittest.TestCase):
    def test_instructions_are_folded_into_one_leading_system_message(self):
        transcript = Transcript.from_messages(
            [
                {"role": "system", "content": "You help."},
                {"role": "system", "content": "Reply in English."},
                {"role": "user", "content": "hello"},
            ]
        )

        messages = transcript.wire()

        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], "You help.\n\nReply in English.")

    def test_consecutive_user_turns_are_joined(self):
        transcript = Transcript.from_messages(
            [
                {"role": "system", "content": "You help."},
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "[framework] return JSON"},
            ]
        )

        messages = transcript.wire()

        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("[framework]", messages[-1]["content"])


class TestEnforcedRules(unittest.TestCase):
    def _error(self, messages) -> str:
        with self.assertRaises(TranscriptError) as ctx:
            Transcript.from_messages(messages).wire()
        return str(ctx.exception)

    def test_a_system_message_after_a_turn_raises(self):
        message = self._error(
            [
                {"role": "system", "content": "You help."},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "Reply in English."},
            ]
        )
        self.assertIn("must come before the conversation starts", message)
        self.assertIn("empty completion", message)

    def test_an_empty_message_raises(self):
        message = self._error(
            [
                {"role": "system", "content": "You help."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "   "},
            ]
        )
        self.assertIn("may not be empty", message)

    def test_two_assistant_turns_raise(self):
        message = self._error(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "assistant", "content": "again"},
            ]
        )
        self.assertIn("Two assistant turns", message)

    def test_a_request_without_a_turn_raises(self):
        message = self._error([{"role": "system", "content": "You help."}])
        self.assertIn("no conversation turn", message)

    def test_a_request_ending_on_assistant_raises(self):
        message = self._error(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        self.assertIn("must end with a user turn", message)

    def test_an_unknown_role_raises(self):
        message = self._error([{"role": "tool", "content": "{}"}])
        self.assertIn("Unknown role", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
