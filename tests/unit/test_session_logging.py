#!/usr/bin/env python3
"""
Tests for chat_workflow.session_logging.

Verifies that get_logs_dir() creates the expected directory and that
log_session() writes correct JSON content to the returned file path.

The filesystem is faked: writing a log is the subject, not the host disk.
"""

import json
import unittest
from pathlib import Path
from typing import override

from pyfakefs.fake_filesystem_unittest import TestCase

from chat_workflow.session_logging import get_logs_dir, log_session


class TestGetLogsDir(TestCase):
    """Tests for get_logs_dir()."""

    @override
    def setUp(self):
        self.setUpPyfakefs()

    def test_get_logs_dir_creates_directory(self):
        """get_logs_dir() must create ~/.chat-workflow/logs/ and return the path."""
        fake_home = Path("/home/test")
        result = get_logs_dir(home=fake_home)

        expected = fake_home / ".chat-workflow" / "logs"
        self.assertEqual(result, expected)
        self.assertTrue(result.exists(), "The logs directory should exist")
        self.assertTrue(result.is_dir(), "The logs directory should be a directory")


class TestLogSession(TestCase):
    """Tests for log_session()."""

    @override
    def setUp(self):
        self.setUpPyfakefs()

    def _sample_messages(self):
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

        from chat_workflow.transcript import dump, load

        return load(
            dump(
                [
                    ModelRequest.user_text_prompt("Hello"),
                    ModelResponse(parts=[TextPart("Hi there")]),
                ]
            )
        )

    def _sample_criteria(self):
        return {
            "budget": {"weight": 8.0, "description": "Budget constraint"},
            "quality": {"weight": 7.0, "description": "Quality level"},
        }

    def _record(self, **overrides):
        from chat_workflow.session_logging import SessionRecord

        values = {
            "messages": self._sample_messages(),
            "criteria": self._sample_criteria(),
            "success_judgement": True,
            "feedback_text": None,
            "model": "test-model",
            "turn_count": 3,
            "context": "test context",
        }
        values.update(overrides)
        return SessionRecord(**values)

    def test_log_session_writes_file(self):
        """log_session() must create a file at the returned path."""
        logs_dir = get_logs_dir()
        result_path = log_session(self._record(), logs_dir=logs_dir)

        self.assertIsInstance(result_path, Path)
        self.assertTrue(result_path.exists(), "The log file should exist on disk")
        self.assertTrue(result_path.is_file(), "The log file should be a regular file")
        self.assertEqual(result_path.parent, logs_dir, "File should be in the logs dir")
        self.assertTrue(
            result_path.name.startswith("session_"),
            "Filename should start with 'session_'",
        )
        self.assertTrue(
            result_path.name.endswith(".json"),
            "Filename should end with '.json'",
        )

    def test_log_session_content(self):
        """log_session() must write valid JSON with correct field values."""
        messages = self._sample_messages()
        criteria = self._sample_criteria()
        result_path = log_session(
            self._record(
                messages=messages,
                criteria=criteria,
                feedback_text="Great session",
                model="gpt-4",
                turn_count=5,
                context="testing content",
            ),
            logs_dir=get_logs_dir(),
        )

        data = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(data["model"], "gpt-4")
        self.assertEqual(data["turn_count"], 5)
        self.assertEqual(data["context"], "testing content")
        from chat_workflow.transcript import dump

        self.assertEqual(data["messages"], json.loads(dump(messages)))
        self.assertEqual(json.loads(dump(messages))[0]["parts"][0]["content"], "Hello")
        self.assertEqual(data["criteria"], criteria)
        self.assertIn("timestamp", data)
        self.assertIn("user_feedback", data)
        self.assertEqual(data["user_feedback"]["success_judgement"], True)
        self.assertEqual(data["user_feedback"]["feedback_text"], "Great session")

    def test_log_session_none_criteria(self):
        """log_session() must handle criteria=None without error."""
        result_path = log_session(
            self._record(
                criteria=None,
                success_judgement=False,
                turn_count=1,
                context="none criteria test",
            ),
            logs_dir=get_logs_dir(),
        )

        data = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertIsNone(data["criteria"])
        self.assertEqual(data["model"], "test-model")
        self.assertEqual(data["turn_count"], 1)
        self.assertEqual(data["context"], "none criteria test")

    def test_log_session_with_feedback(self):
        """log_session() must include feedback_text in the output when provided."""
        feedback = "The model needs improvement on budget estimation."
        result_path = log_session(
            self._record(
                success_judgement=False,
                feedback_text=feedback,
                model="claude-3",
                turn_count=7,
                context="feedback test",
            ),
            logs_dir=get_logs_dir(),
        )

        data = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(data["user_feedback"]["success_judgement"], False)
        self.assertEqual(data["user_feedback"]["feedback_text"], feedback)
        self.assertEqual(data["model"], "claude-3")
        self.assertEqual(data["turn_count"], 7)
        self.assertEqual(data["context"], "feedback test")

    def test_log_session_empty_messages(self):
        """log_session() must handle an empty messages list."""
        result_path = log_session(
            self._record(
                messages=[],
                criteria=None,
                turn_count=0,
                model="empty-test",
                context="empty messages",
            ),
            logs_dir=get_logs_dir(),
        )

        data = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(data["messages"], [])
        self.assertEqual(data["turn_count"], 0)
        self.assertEqual(data["model"], "empty-test")

    def test_log_session_timestamp_format(self):
        """The timestamp field must be a valid ISO-8601 string."""
        from datetime import datetime

        result_path = log_session(
            self._record(turn_count=2, model="ts-test", context="timestamp check"),
            logs_dir=get_logs_dir(),
        )

        data = json.loads(result_path.read_text(encoding="utf-8"))

        parsed = datetime.fromisoformat(data["timestamp"])
        self.assertIsInstance(parsed, datetime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
