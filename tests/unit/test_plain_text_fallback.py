"""A plain-text assistant turn is a continue, not a validation failure."""

import unittest
from types import SimpleNamespace

from chat_workflow.atomic_workflow_step import _plain_text_completion


class FakeInstructorRetryError(Exception):
    """The attributes ``_plain_text_completion`` reads from Instructor's error."""

    def __init__(self, content: str):
        super().__init__(content)
        self.last_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        self.failed_attempts = []


class TestPlainTextFallback(unittest.TestCase):
    def test_plain_text_is_accepted(self):
        exc = FakeInstructorRetryError(
            "Almost there. One last thing — where do you keep it?"
        )

        self.assertEqual(
            _plain_text_completion(exc),
            "Almost there. One last thing — where do you keep it?",
        )

    def test_malformed_json_is_not_plain_text(self):
        exc = FakeInstructorRetryError('{"intent": "continue", "message": "unterminated')

        self.assertIsNone(_plain_text_completion(exc))

    def test_bare_list_is_not_plain_text(self):
        exc = FakeInstructorRetryError('[{"name": "minutes"}]')

        self.assertIsNone(_plain_text_completion(exc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
