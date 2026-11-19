"""Tests for the recursive Session and the SessionComponent wrapper.

The wrapper is explicit: it greets only when the session has no messages, runs
the step, then signs off. A child session starts with clean context and records
its own messages; the parent records the child.
"""

import unittest
from inspect import cleandoc
from typing import override

from chat_workflow import Session, SessionComponent, SessionLog, TextReply
from chat_workflow.session import UserIO
from tests.conftest import FakeConfig


class _IO(UserIO):
    def __init__(self) -> None:
        self.echoes: list[str] = []
        self.prompts: list[str] = []

    @override
    def echo(self, message: str) -> None:
        self.echoes.append(message)

    @override
    def prompt(self, label: str) -> str:
        self.prompts.append(label)
        return "an answer"


def _session() -> Session:
    return Session(io=_IO(), state=SessionLog(), config=FakeConfig())


class TestRecursiveSession(unittest.TestCase):
    def test_spawn_creates_a_child_with_clean_context(self):
        root = _session()
        root.state.messages.append(object())  # type: ignore[arg-type]

        child = root.spawn(title="Meeting", instructions="the brief")

        self.assertIs(child.parent, root)
        self.assertEqual(root.children, [child])
        self.assertEqual(child.state.messages, [])
        self.assertEqual(child.instructions, "the brief")
        self.assertIs(child.raw_io, root.raw_io)
        self.assertIs(child.config, root.config)
        self.assertEqual(child.title, "Meeting")
        self.assertIsNot(child.state, root.state)

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

    def test_say_appends_an_assistant_turn_and_echoes(self):
        io = _IO()
        session = Session(io=io, state=SessionLog(), config=FakeConfig())

        session.say("Hi there!")

        self.assertEqual(len(session.state.messages), 1)
        self.assertTrue(any("Hi there!" in echo for echo in io.echoes))


class TestSessionComponent(unittest.TestCase):
    def test_converse_greets_runs_the_step_and_signs_off(self):
        session = _session()
        seen: dict[str, bool] = {}

        def step(*, session: Session) -> str:
            seen["active"] = session.active
            session.state.messages.append(object())  # type: ignore[arg-type]
            return "the result"

        original_greet = SessionComponent.greet
        original_present = SessionComponent.present
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: TextReply(  # type: ignore[method-assign]
            text="Hi there! I help describe your work."
        )
        SessionComponent.present = lambda self, *, session, result, **kwargs: (  # type: ignore[method-assign]
            TextReply(text="Here is the result.")
        )
        SessionComponent.sign_off = lambda self, *, session, result=None, **kwargs: (  # type: ignore[method-assign]
            TextReply(text="I have what I need.")
        )
        try:
            result = SessionComponent().converse(step, session=session)
        finally:
            SessionComponent.greet = original_greet  # type: ignore[method-assign]
            SessionComponent.present = original_present  # type: ignore[method-assign]
            SessionComponent.sign_off = original_sign_off  # type: ignore[method-assign]

        self.assertEqual(result, "the result")
        self.assertTrue(seen["active"])
        self.assertFalse(session.active)
        self.assertEqual(
            session.instructions,
            cleandoc(SessionComponent.converse.__doc__ or ""),
        )
        self.assertEqual(session.state.final_result, "the result")

    def test_converse_does_not_greet_a_session_that_has_messages(self):
        session = _session()
        session.state.messages.append(object())  # type: ignore[arg-type]
        calls: list[str] = []

        original_greet = SessionComponent.greet
        original_present = SessionComponent.present
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: calls.append("greet")  # type: ignore[method-assign]
        SessionComponent.present = lambda self, *, session, result, **kwargs: (  # type: ignore[method-assign]
            TextReply(text="Here is the result.")
        )
        SessionComponent.sign_off = lambda self, *, session, result=None, **kwargs: (  # type: ignore[method-assign]
            TextReply(text="I have what I need.")
        )
        try:
            SessionComponent().converse(lambda *, session: "result", session=session)
        finally:
            SessionComponent.greet = original_greet  # type: ignore[method-assign]
            SessionComponent.present = original_present  # type: ignore[method-assign]
            SessionComponent.sign_off = original_sign_off  # type: ignore[method-assign]

        self.assertEqual(calls, [])
        # The seeded message, the presentation the step did not make, and the
        # closing.
        self.assertEqual(len(session.state.messages), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
