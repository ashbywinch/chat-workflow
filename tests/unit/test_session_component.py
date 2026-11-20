"""The recursive Session and the SessionComponent wrapper.

The wrapper greets only when the session has no messages, runs the step, then
signs off. It never speaks for the step: a conversational step's success
message is the model's own words, spoken by ``Session.run``.
"""

import unittest
from inspect import cleandoc
from typing import override

from chat_workflow import Session, SessionComponent, SessionLog, TextReply
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.session import UserIO
from chat_workflow.transcript import entries
from tests.conftest import (
    FakeConfig,
    make_atomic_workflow_config,
    make_conversation,
    make_valid_criteria,
)


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

    def test_propose_renders_and_says_a_non_empty_result(self):
        session = _session()

        from workflows.workflow.models.process_definition import (
            ProcessDefinition,
        )

        session.propose(
            ProcessDefinition(
                phases=["Capture"],
                activities=["Scribble"],
                participants=["You"],
                orchestrating_component="You",
            )
        )

        self.assertTrue(any("Stages: Capture" in str(m) for m in session.state.messages))

    def test_propose_says_nothing_for_an_empty_list(self):
        session = _session()

        session.propose([])

        self.assertEqual(session.state.messages, [])


class TestTheSuccessMessage(unittest.TestCase):
    """A conversational step's completion is something the user reads."""

    def test_a_chat_on_step_speaks_its_success_message(self):
        session = _session()
        conversation = make_conversation(
            [
                {
                    "intent": "success",
                    "message": "Here is what I put together.",
                    "result": make_valid_criteria().model_dump(),
                }
            ]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        session.run(workflow, "", chat=True)

        kinds = entries(session.state.messages)
        self.assertIn(("assistant", "Here is what I put together."), kinds)

    def test_a_chat_off_step_does_not_speak_it(self):
        session = _session()
        conversation = make_conversation(
            [
                {
                    "intent": "success",
                    "message": "Internal summary only.",
                    "result": make_valid_criteria().model_dump(),
                }
            ]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation)
        )

        session.run(workflow, "", chat=False)

        self.assertNotIn(
            ("assistant", "Internal summary only."),
            entries(session.state.messages),
        )


class TestSessionComponent(unittest.TestCase):
    def test_converse_greets_runs_the_step_and_signs_off(self):
        session = _session()
        seen: dict[str, bool] = {}

        def step(*, session: Session) -> str:
            seen["active"] = session.active
            session.state.messages.append(object())  # type: ignore[arg-type]
            return "the result"

        original_greet = SessionComponent.greet
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: TextReply(  # type: ignore[method-assign]
            text="Hi there! I help describe what you do."
        )
        SessionComponent.sign_off = lambda self, *, session, **kwargs: (  # type: ignore[method-assign]
            TextReply(text="I have what I need.")
        )
        try:
            result = SessionComponent().converse(step, session=session)
        finally:
            SessionComponent.greet = original_greet  # type: ignore[method-assign]
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
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: calls.append("greet")  # type: ignore[method-assign]
        SessionComponent.sign_off = lambda self, *, session, **kwargs: TextReply(text="Done.")  # type: ignore[method-assign]
        try:
            SessionComponent().converse(lambda *, session: "result", session=session)
        finally:
            SessionComponent.greet = original_greet  # type: ignore[method-assign]
            SessionComponent.sign_off = original_sign_off  # type: ignore[method-assign]

        self.assertEqual(calls, [])

    def test_a_chat_off_step_has_no_greeting_or_closing(self):
        session = _session()
        called: list[str] = []
        original_greet = SessionComponent.greet
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: called.append("greet")  # type: ignore[method-assign]
        SessionComponent.sign_off = lambda self, *, session, **kwargs: called.append("close")  # type: ignore[method-assign]

        def step(*, session):
            session.state.messages.append(object())  # type: ignore[arg-type]
            return "done"

        step._chat = False  # type: ignore[attr-defined]
        try:
            self.assertEqual(SessionComponent().converse(step, session=session), "done")
        finally:
            SessionComponent.greet = original_greet  # type: ignore[method-assign]
            SessionComponent.sign_off = original_sign_off  # type: ignore[method-assign]

        self.assertEqual(called, [])

    def test_a_failed_greeting_falls_back(self):
        session = _session()
        original_greet = SessionComponent.greet
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: (_ for _ in ()).throw(RuntimeError("no route"))  # type: ignore[method-assign]
        SessionComponent.sign_off = lambda self, *, session, **kwargs: TextReply(text="Done.")  # type: ignore[method-assign]
        try:
            SessionComponent().converse(lambda *, session: "the result", session=session)
        finally:
            SessionComponent.greet = original_greet  # type: ignore[method-assign]
            SessionComponent.sign_off = original_sign_off  # type: ignore[method-assign]

        self.assertTrue(
            any("what are you working on" in str(m) for m in session.state.messages)
        )

    def test_a_failed_or_questioning_closing_is_a_statement(self):
        session = _session()
        original_greet = SessionComponent.greet
        original_sign_off = SessionComponent.sign_off
        SessionComponent.greet = lambda self, *, session, **kwargs: TextReply(text="Hi.")  # type: ignore[method-assign]

        for closing in (
            lambda self, *, session, **kwargs: (_ for _ in ()).throw(RuntimeError("no route")),
            lambda self, *, session, **kwargs: TextReply(text="Anything else?"),
        ):
            SessionComponent.sign_off = closing  # type: ignore[method-assign]
            try:
                SessionComponent().converse(lambda *, session: "the result", session=session)
            finally:
                pass

        SessionComponent.greet = original_greet  # type: ignore[method-assign]
        SessionComponent.sign_off = original_sign_off  # type: ignore[method-assign]

        self.assertTrue(
            any("I have what I need" in str(m) for m in session.state.messages)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
