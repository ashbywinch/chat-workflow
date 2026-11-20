"""A step declares the minimum exchanges its task needs.

A gather step that must ask before it proposes cannot finish on its first
turn; the framework feeds the early finish back as a correction instead of
accepting a silent summary.
"""

import unittest

from chat_workflow.atomic_workflow import AtomicWorkflow
from tests.conftest import (
    make_atomic_workflow_config,
    make_conversation,
    make_valid_criteria,
)


class TestMinTurns(unittest.TestCase):
    def _success(self) -> dict:
        return {
            "intent": "success",
            "message": "Done.",
            "result": make_valid_criteria().model_dump(),
        }

    def test_an_early_success_is_a_correction(self):
        conversation = make_conversation([self._success(), self._success()])
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(
                conversation=conversation, min_turns=2
            )
        )

        result = workflow.process_turn("go")

        self.assertTrue(result.is_complete)
        self.assertEqual(len(conversation.script.calls), 2)

    def test_the_default_allows_a_first_turn_success(self):
        conversation = make_conversation([self._success()])
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        result = workflow.process_turn("go")

        self.assertTrue(result.is_complete)
        self.assertEqual(len(conversation.script.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
