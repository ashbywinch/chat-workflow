"""Tests for chat-off atomic workflow steps."""

import unittest
from typing import cast, override

from chat_workflow import AtomicWorkflowStep, Session, SessionLog, TurnResult
from chat_workflow.session import UserIO


class _NoPromptIO(UserIO):
    def __init__(self):
        self.echoes: list[str] = []
        self.prompts: list[str] = []

    @override
    def echo(self, message: str) -> None:
        self.echoes.append(message)

    @override
    def prompt(self, label: str, **kwargs) -> str:
        self.prompts.append(label)
        raise AssertionError("chat=False must never prompt the user")


class _ChatOffWorkflow:
    def __init__(self):
        self.chat = False
        self.model = "test-model"
        self.messages = []
        self.turn_count = 0

    def process_turn(self, user_input: str):
        self.turn_count += 1
        if self.turn_count == 1:
            return TurnResult.continuing("I need more information")
        return TurnResult.success("done")


class TestChatOffWorkflow(unittest.TestCase):
    def test_chat_off_never_prompts_and_uses_in_context_correction(self):
        io = _NoPromptIO()
        workflow = _ChatOffWorkflow()
        session = Session(io=io, state=SessionLog())

        result = session.run(workflow_step=cast(AtomicWorkflowStep[str], workflow), first_user_input="")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, "done")
        self.assertEqual(io.prompts, [])
        self.assertEqual(workflow.turn_count, 2)
        correction = workflow.messages[-1]["content"]
        self.assertIn("No user is available", correction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
