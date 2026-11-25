"""The recursive Session and the SessionComponent wrapper.

The wrapper greets only when the session has no messages, runs the step, then
signs off. It never speaks for the step: a conversational step's success
message is the model's own words, spoken by ``Session.run``.
"""

import unittest
from collections.abc import Callable
from inspect import cleandoc
from typing import override

from chat_workflow import Session, SessionComponent, SessionLog, TextReply
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.exceptions import MaxRetriesExceededError
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


class _Draft:
    """A minimal Renderable: what the user sees is its text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, context) -> str:
        return self.text


class TestRecursiveSession(unittest.TestCase):
    def test_spawn_creates_a_child_with_clean_context(self):
        root = _session()
        root.say("an earlier turn")

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
            ProcessStage,
        )

        session.propose(
            ProcessDefinition(
                stages=[ProcessStage(name="Capture", steps=["Scribble"])],
                participants=["You"],
                orchestrating_component="You",
            )
        )

        self.assertTrue(
            any("1 — Capture" in str(m) for m in session.state.messages)
        )

    def test_propose_says_nothing_for_an_empty_list(self):
        session = _session()

        session.propose([])

        self.assertEqual(session.state.messages, [])

    def test_a_proposal_is_one_turn_with_one_question(self):
        session = _session()

        session.propose([_Draft("first"), _Draft("second")], note="The rest stays as we agreed.")

        text = str(session.state.messages[-1])
        self.assertEqual(len(session.state.messages), 1)
        self.assertIn("Here's what I've got so far", text)
        self.assertIn("first", text)
        self.assertIn("The rest stays as we agreed.", text)
        self.assertEqual(text.count("?"), 1)

    def test_a_changed_proposal_asks_about_the_change(self):
        session = _session()

        session.propose([_Draft("first")], changed=True)

        text = str(session.state.messages[-1])
        self.assertIn("Here's what I've changed", text)
        self.assertIn("Is that the change you meant?", text)


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

        session.run(workflow, chat=True)

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

        session.run(workflow, chat=False)

        self.assertNotIn(
            ("assistant", "Internal summary only."),
            entries(session.state.messages),
        )

    def test_a_chat_off_step_cannot_continue(self):
        """The chat-off output shape has no continue, so a question is rejected."""
        session = _session()
        conversation = make_conversation(
            [{"intent": "continue", "message": "Which format?"}]
        )
        workflow = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=conversation),
            chat=False,
        )

        with self.assertRaises(MaxRetriesExceededError):
            session.run(workflow, chat=False)

        # The initial request plus the output retries; the schema never accepts
        # a continue.
        self.assertEqual(len(conversation.script.calls), 3)


def _no_route() -> TextReply:
    """A model call that cannot reach the provider."""
    raise RuntimeError("no route")


class _StubComponent(SessionComponent):
    """An object fake for the cosmetic open and close.

    ``converse`` is the subject; the open and the close are model calls, so a
    test hands them fixed text (or a failure) and reads ``called`` to see which
    of the two ran.
    """

    def __init__(
        self,
        *,
        greeting: Callable[[], TextReply] | None = None,
        closing: Callable[[], TextReply] | None = None,
    ) -> None:
        self.called: list[str] = []
        self._greeting = greeting or (lambda: TextReply(text="Hi there! I help describe what you do."))
        self._closing = closing or (lambda: TextReply(text="I have what I need."))

    @override
    def greet(self, *, session: Session, max_turns: int = 1) -> TextReply:
        self.called.append("greet")
        return self._greeting()

    @override
    def sign_off(self, *, session: Session, max_turns: int = 1) -> TextReply:
        self.called.append("close")
        return self._closing()


class _ChatOffStep:
    """A step with no user of its own: ``converse`` reads ``_chat`` off it."""

    _chat = False

    def __call__(self, *, session: Session) -> str:
        session.say("a step with nothing to ask")
        return "done"


class TestSessionComponent(unittest.TestCase):
    def test_converse_greets_runs_the_step_and_signs_off(self):
        session = _session()
        component = _StubComponent()
        seen: dict[str, bool] = {}

        def step(*, session: Session) -> str:
            seen["active"] = session.active
            session.say("a turn the step took")
            return "the result"

        result = component.converse(step, session=session)

        self.assertEqual(result, "the result")
        self.assertTrue(seen["active"])
        self.assertEqual(component.called, ["greet", "close"])
        self.assertFalse(session.active)
        self.assertEqual(
            session.instructions,
            cleandoc(SessionComponent.converse.__doc__ or ""),
        )
        self.assertEqual(session.state.final_result, "the result")

    def test_converse_does_not_greet_a_session_that_has_messages(self):
        session = _session()
        session.say("an earlier turn")
        component = _StubComponent()

        component.converse(lambda *, session: "result", session=session)

        self.assertEqual(component.called, ["close"])

    def test_a_chat_off_step_has_no_greeting_or_closing(self):
        session = _session()
        component = _StubComponent()

        self.assertEqual(component.converse(_ChatOffStep(), session=session), "done")

        self.assertEqual(component.called, [])

    def test_a_failed_greeting_falls_back(self):
        session = _session()
        component = _StubComponent(greeting=_no_route)

        component.converse(lambda *, session: "the result", session=session)

        self.assertTrue(
            any("here to help you describe" in str(m) for m in session.state.messages)
        )

    def test_a_failed_or_questioning_closing_is_a_statement(self):
        for closing in (
            _no_route,
            lambda: TextReply(text="Anything else?"),
        ):
            with self.subTest(closing=closing):
                session = _session()
                component = _StubComponent(closing=closing)

                component.converse(lambda *, session: "the result", session=session)

                self.assertTrue(
                    any("I have what I need" in str(m) for m in session.state.messages)
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
