"""Tests for the recursive Session and the SessionComponent wrapper.

The wrapper is explicit: it greets only when the session has no messages, runs
the step, then signs off. A child session starts with clean context and records
its own messages; the parent records the child as an event.
"""

import unittest
from inspect import cleandoc
from typing import cast, override
from unittest.mock import patch

from pydantic import BaseModel

from chat_workflow import (
    AgentIntent,
    AgentResponse,
    AtomicWorkflowStep,
    AtomicWorkflowStepConfig,
    Session,
    SessionComponent,
    SessionLog,
    TurnResult,
)
from chat_workflow.session import UserIO
from chat_workflow.session_log import _record_atomic_workflow_step
from tests.conftest import FakeConfig


class _IO(UserIO):
    def __init__(self) -> None:
        self.echoes: list[str] = []
        self.prompts: list[str] = []

    @override
    def echo(self, message: str) -> None:
        self.echoes.append(message)

    @override
    def prompt(self, label: str, **kwargs) -> str:
        self.prompts.append(label)
        return "an answer"


def _session() -> Session:
    return Session(io=_IO(), state=SessionLog(), config=FakeConfig())


class TestRecursiveSession(unittest.TestCase):
    def test_spawn_creates_a_child_with_clean_context(self):
        root = _session()
        root.state.messages.append({"role": "user", "content": "the brief"})

        child = root.spawn(title="Meeting", instructions="the brief")

        self.assertIs(child.parent, root)
        self.assertEqual(root.children, [child])
        self.assertEqual(child.state.messages, [])
        self.assertEqual(child.instructions, "the brief")
        self.assertIs(child.raw_io, root.raw_io)
        self.assertIs(child.config, root.config)
        self.assertEqual(child.title, "Meeting")
        self.assertIsNot(child.state, root.state)

    def test_spawn_records_a_child_session_event_on_the_parent(self):
        root = _session()

        child = root.spawn(title="Meeting")

        event = root.state.events[-1]
        self.assertEqual(event.kind, "child_session")
        self.assertEqual(event.meta["session"], child.id)
        self.assertEqual(event.meta["title"], "Meeting")
        self.assertEqual(event.meta["order"], 1)

    def test_walk_returns_the_tree_in_spawn_order(self):
        root = _session()
        first = root.spawn(title="First")
        second = root.spawn(title="Second")
        grandchild = first.spawn(title="Grandchild")

        self.assertEqual(
            [item.id for item in root.walk()],
            [root.id, first.id, grandchild.id, second.id],
        )
        self.assertIs(root.child(grandchild.id), grandchild)


class TestSessionComponent(unittest.TestCase):
    @patch("chat_workflow.atomic_workflow_step.AtomicWorkflowStep._call_llm")
    def test_converse_greets_runs_the_step_and_signs_off(self, mock_call_llm):
        mock_call_llm.side_effect = [
            AgentResponse[str](
                intent=AgentIntent.SUCCESS,
                result="Hi there! I help describe your work. What would you like help with?",
            ),
            AgentResponse[str](
                intent=AgentIntent.SUCCESS,
                result="Great, I have what I need.",
            ),
        ]
        session = _session()
        seen: dict[str, bool] = {}

        def step(*, session: Session) -> str:
            seen["active"] = session.active
            session.state.messages.append({"role": "user", "content": "my minutes"})
            session.state.messages.append({"role": "assistant", "content": "Got it."})
            session.state.messages.append({"role": "user", "content": "yes"})
            return "the result"

        result = SessionComponent().converse(step, session=session)

        self.assertEqual(result, "the result")
        self.assertTrue(seen["active"])
        self.assertFalse(session.active)
        self.assertEqual(
            session.instructions,
            cleandoc(SessionComponent.converse.__doc__ or ""),
        )
        self.assertEqual(session.state.messages[0]["role"], "assistant")
        self.assertIn("Hi there!", session.state.messages[0]["content"])
        self.assertEqual(
            session.state.messages[-1]["content"],
            "Great, I have what I need.",
        )
        self.assertEqual(session.state.final_result, "the result")

    @patch("chat_workflow.atomic_workflow_step.AtomicWorkflowStep._call_llm")
    def test_converse_does_not_greet_a_session_that_has_messages(self, mock_call_llm):
        mock_call_llm.side_effect = [
            AgentResponse[str](
                intent=AgentIntent.SUCCESS,
                result="Great, I have what I need.",
            )
        ]
        session = _session()
        session.state.messages.append({"role": "user", "content": "continue"})

        SessionComponent().converse(
            lambda *, session: "result",
            session=session,
        )

        self.assertEqual(mock_call_llm.call_count, 1)
        self.assertNotIn("Hi there!", session.state.messages[0]["content"])

    @patch("chat_workflow.atomic_workflow_step.AtomicWorkflowStep._call_llm")
    def test_converse_does_not_wrap_a_chat_off_step(self, mock_call_llm):
        session = _session()

        def step(*, session: Session) -> str:
            step._chat = False  # type: ignore[attr-defined]
            return "quiet"

        step._chat = False  # type: ignore[attr-defined]

        self.assertEqual(
            SessionComponent().converse(step, session=session),
            "quiet",
        )
        self.assertEqual(mock_call_llm.call_count, 0)
        self.assertEqual(session.state.messages, [])
        self.assertFalse(session.active)


class _Result(BaseModel):
    x: str


class TestRecordAtomicWorkflowStep(unittest.TestCase):
    def test_system_prompt_and_framework_turns_are_not_session_history(self):
        config = AtomicWorkflowStepConfig(
            system_prompt="SYSTEM PROMPT",
            response_model=AgentResponse[_Result],
            max_turns=3,
            on_continue=lambda response: TurnResult[_Result].continuing(
                response.message or ""
            ),
            on_success=lambda response: TurnResult[_Result].success(
                cast(_Result, response.result)
            ),
            on_failure=lambda response: RuntimeError(response.message or "failed"),
        )
        step = AtomicWorkflowStep(config=config)
        log = SessionLog()

        with patch.object(
            AtomicWorkflowStep,
            "_call_llm",
            return_value=AgentResponse[_Result](
                intent=AgentIntent.CONTINUE, message="What next?"
            ),
        ):
            step.process_turn("hello")

        _record_atomic_workflow_step(log, step)

        self.assertEqual(
            log.messages,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "What next?"},
            ],
        )
        self.assertFalse(any(message["role"] == "system" for message in log.messages))


class TestSingleTurnStep(unittest.TestCase):
    def test_continue_message_is_the_result_without_a_user_turn(self):
        config = AtomicWorkflowStepConfig(
            system_prompt="SYSTEM PROMPT",
            response_model=AgentResponse[str],
            max_turns=1,
            chat=False,
            single_turn=True,
            on_continue=lambda response: TurnResult[str].continuing(
                response.message or ""
            ),
            on_success=lambda response: TurnResult[str].success(response.result or ""),
            on_failure=lambda response: RuntimeError(response.message or "failed"),
        )
        step = AtomicWorkflowStep(config=config)

        with patch.object(
            AtomicWorkflowStep,
            "_call_llm",
            return_value=AgentResponse[str](
                intent=AgentIntent.CONTINUE, message="Hi there!"
            ),
        ):
            result = step.process_turn("")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, "Hi there!")
        self.assertEqual(step.messages, [{"role": "system", "content": "SYSTEM PROMPT"}])


class TestSessionComponentContract(unittest.TestCase):
    """The framework's own session component follows the rules it enforces."""

    def test_converse_docstring_is_the_overall_conversation_instructions(self):
        doc = cleandoc(SessionComponent.converse.__doc__ or "")

        self.assertTrue(doc.strip())
        self.assertIn("their words", doc)
        lowered = doc.lower()
        for phrase in ("return json", "put the text in `message`", "set `intent`"):
            self.assertNotIn(phrase, lowered)

    def test_greeting_and_closing_are_single_turn_chat_off_steps(self):
        for step in (SessionComponent.greet, SessionComponent.sign_off):
            self.assertTrue(step._is_workflow_step)  # type: ignore[attr-defined]
            self.assertFalse(step._chat)  # type: ignore[attr-defined]
            self.assertTrue(step._single_turn)  # type: ignore[attr-defined]
            self.assertTrue(cleandoc(step.__doc__ or "").strip())

    def test_the_checklist_is_declared_on_the_step_and_is_the_judge_list(self):
        from chat_workflow.conversation_rules import DEFAULT_RULES
        from chat_workflow.decorators import atomic_workflow_step

        @atomic_workflow_step(conversation_validation_rules=[("Author rule", "Do not.")])
        def step(*, session: Session) -> str:
            """A conversation step."""
            ...

        rules = step.__conversation_rules__  # type: ignore[attr-defined]
        self.assertTrue(set(DEFAULT_RULES).issubset(set(rules.items())))
        self.assertEqual(rules["Author rule"], "Do not.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
