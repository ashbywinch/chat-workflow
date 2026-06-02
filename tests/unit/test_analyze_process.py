"""Tests for generate_from_chat composite workflow."""

import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from workflows.workflow.models import ProcessDefinition, generate_from_chat


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


def _make_session() -> Session:
    return Session(
        io=MagicMock(),
        state=SessionLog(),
        config=FakeConfig(),
    )


class TestGenerateFromChat(unittest.TestCase):
    def test_has_workflow_attribute(self):
        self.assertTrue(
            getattr(generate_from_chat, "_is_workflow", False),
            "generate_from_chat should have _is_workflow=True",
        )

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            generate_from_chat()
        self.assertIn("session", str(ctx.exception))

    @patch("workflows.workflow.models.process_definition._generate_from_notes")
    @patch("workflows.workflow.models.process_definition._gather_notes")
    def test_orchestrates_gather_then_generate(
        self, mock_gather, mock_generate
    ):
        mock_gather.return_value = "Some raw notes about the process"
        mock_generate.return_value = ProcessDefinition(
            phases=["Plan", "Do", "Review"],
            activities=["Plan work", "Execute", "Review results"],
            orchestrating_component="Manager",
            participants=["Manager", "Team"],
        )
        session = _make_session()
        result = generate_from_chat(session=session)
        mock_gather.assert_called_once_with(session=session, max_turns=10)
        mock_generate.assert_called_once_with(
            notes="Some raw notes about the process",
            session=session,
            max_turns=10,
        )
        self.assertIsInstance(result, ProcessDefinition)
        self.assertEqual(result.orchestrating_component, "Manager")

    @patch("workflows.workflow.models.process_definition._generate_from_notes")
    @patch("workflows.workflow.models.process_definition._gather_notes")
    def test_passes_max_turns(
        self, mock_gather, mock_generate
    ):
        mock_gather.return_value = "notes"
        mock_generate.return_value = ProcessDefinition(
            phases=["Test"],
            activities=["Test"],
            orchestrating_component="Test",
            participants=["Test"],
        )
        session = _make_session()
        generate_from_chat(session=session, max_turns=5)
        mock_gather.assert_called_once_with(session=session, max_turns=5)
        mock_generate.assert_called_once_with(
            notes="notes", session=session, max_turns=5
        )
