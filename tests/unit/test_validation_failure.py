"""The validation-failure classifier maps known failure shapes to guidance."""

import unittest
from json import JSONDecodeError

from instructor.core.exceptions import InstructorRetryException
from instructor.core.retry import FailedAttempt
from pydantic import ValidationError as PydanticValidationError

from chat_workflow.atomic_workflow_step import (
    ValidationFailure,
    _correction_prompt,
    _validation_failure,
)


def _retry(cause: Exception) -> InstructorRetryException:
    return InstructorRetryException(
        "rejected",
        n_attempts=1,
        total_usage=0,
        failed_attempts=[FailedAttempt(1, cause)],
    )


class TestValidationFailureClassifier(unittest.TestCase):
    def test_plain_text_gets_the_envelope_guidance(self):
        exc = _retry(
            JSONDecodeError(
                "Invalid JSON: expected value at line 1 column 1",
                "Almost there. One last thing?",
                0,
            )
        )

        failure = _validation_failure(exc)

        assert failure is not None
        self.assertIn("plain text instead of JSON", failure.message())
        self.assertIn('"intent": "continue"', failure.message())

    def test_malformed_json_gets_escaping_guidance(self):
        exc = _retry(
            JSONDecodeError(
                "Invalid JSON: control character found while parsing a string",
                '{"message": "line\nbreak"}',
                10,
            )
        )

        failure = _validation_failure(exc)

        assert failure is not None
        self.assertIn("Encode every newline", failure.message())

    def test_pydantic_failure_is_a_correction(self):
        exc = _retry(PydanticValidationError.from_exception_data("Model", []))

        failure = _validation_failure(exc)

        assert failure is not None
        self.assertEqual(failure.kind, "validation_correction")

    def test_non_validation_failure_is_not_retried(self):
        exc = _retry(ConnectionError("provider is down"))

        self.assertIsNone(_validation_failure(exc))

    def test_language_correction_uses_its_own_prompt(self):
        failure = ValidationFailure(
            reason="Reply in the user's language.",
            kind="language_correction",
        )

        prompt = _correction_prompt(failure)

        self.assertEqual(prompt, "[framework] Reply in the user's language.")
        self.assertNotIn("Fix exactly these violations", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
