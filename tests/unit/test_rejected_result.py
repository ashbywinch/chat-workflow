"""A result its own rules reject is a turn, not the end of the run.

Every output retry answers from the same history: when a rule fails for a
fact the user has not given, resampling cannot supply it. The rejection is
fed back as a correction turn, so the model asks for what is missing and the
user gets to answer instead of losing the conversation.
"""

from __future__ import annotations

import unittest
from typing import Any

from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.exceptions import MaxRetriesExceededError
from tests.conftest import (
    make_atomic_workflow_config,
    make_conversation,
    make_valid_criteria,
)

# A criterion list without the budget the answer model requires: a rule
# failure, not a shape failure, and one no resample of the same history fixes.
_BUDGETLESS = {
    "intent": "success",
    "result": {
        "context": "A birthday present",
        "criteria": [
            {"name": "price", "description": "Keeps the gift affordable"},
            {"name": "thought", "description": "Shows the person was considered"},
        ],
    },
}
_ASKS_FOR_BUDGET = {"intent": "continue", "message": "What is your budget?"}


def _request_text(messages: list[Any]) -> str:
    """Everything the model was told in one request, as plain text."""
    return " ".join(
        part.content
        for message in messages
        for part in message.parts
        if isinstance(getattr(part, "content", None), str)
    )


class TestRejectedResult(unittest.TestCase):
    def test_a_rejected_result_asks_the_user_instead_of_ending_the_run(self):
        conversation = make_conversation(
            [_BUDGETLESS, _ASKS_FOR_BUDGET],
            output_retries=0,
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        result = workflow.process_turn("I need a birthday present")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What is your budget?")
        self.assertEqual(len(conversation.script.calls), 2)
        correction = _request_text(conversation.script.calls[1])
        # The rule that was broken reaches the model, not only a retry count.
        self.assertIn("budget", correction.lower())
        # And the correction says where the missing fact comes from.
        self.assertIn("Ask for what is still missing", correction)
        self.assertIn("do not invent", correction)

    def test_the_last_turn_still_fails_rather_than_asking(self):
        conversation = make_conversation([_BUDGETLESS], output_retries=0)
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(
                conversation=conversation, max_turns=1
            )
        )

        with self.assertRaises(MaxRetriesExceededError):
            workflow.process_turn("go")

        self.assertEqual(len(conversation.script.calls), 1)

    def test_a_rejection_that_the_next_turn_completes(self):
        conversation = make_conversation(
            [
                _BUDGETLESS,
                {"intent": "continue", "message": "What is your budget?"},
                {
                    "intent": "success",
                    "result": make_valid_criteria().model_dump(),
                },
            ],
            output_retries=0,
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        first = workflow.process_turn("I need a birthday present")
        second = workflow.process_turn("About fifty pounds")

        self.assertFalse(first.is_complete)
        self.assertTrue(second.is_complete)
        self.assertEqual(len(conversation.script.calls), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
