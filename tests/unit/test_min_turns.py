"""A step declares the minimum exchanges its task needs.

A gather step that must ask before it proposes cannot finish on its first
turn; the framework feeds the early finish back as a correction instead of
accepting a silent summary.
"""

import unittest

from chat_workflow import AtomicWorkflowFailedError
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


class TestTurnShape(unittest.TestCase):
    """A turn says what its shape can say, and does not repeat itself."""

    def test_an_identical_continue_is_returned_as_is(self):
        """There is no runtime repetition guard: the judge reads the transcript."""
        conversation = make_conversation(
            [
                {"intent": "continue", "message": "What do you make?"},
                {"intent": "continue", "message": "What do you make?"},
            ]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        first = workflow.process_turn("go")
        second = workflow.process_turn("a list")

        self.assertEqual(first.message, "What do you make?")
        self.assertEqual(second.message, "What do you make?")
        self.assertEqual(len(conversation.script.calls), 2)

    def test_a_continue_that_leads_into_content_is_returned_as_is(self):
        """A lead-in is a content defect, not a text shape the runtime can
        interpret; the contract is in the prompt."""
        conversation = make_conversation(
            [{"intent": "continue", "message": "Here's the structure:"}]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        result = workflow.process_turn("go")

        self.assertEqual(result.message, "Here's the structure:")
        self.assertEqual(len(conversation.script.calls), 1)

    def test_a_failure_is_given_one_turn_to_continue(self):
        conversation = make_conversation(
            [
                {"intent": "failure", "message": "I cannot do this."},
                {
                    "intent": "success",
                    "result": make_valid_criteria().model_dump(),
                },
            ]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        result = workflow.process_turn("go")

        self.assertTrue(result.is_complete)
        self.assertEqual(len(conversation.script.calls), 2)

    def test_a_second_failure_stops(self):
        conversation = make_conversation(
            [
                {"intent": "failure", "message": "I cannot do this."},
                {"intent": "failure", "message": "I still cannot."},
            ]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        with self.assertRaises(AtomicWorkflowFailedError):
            workflow.process_turn("go")

        self.assertEqual(len(conversation.script.calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
