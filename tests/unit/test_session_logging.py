#!/usr/bin/env python3
"""
Tests for chat_workflow.session_logging.

Verifies that get_logs_dir() creates the expected directory and that
log_session() writes correct JSON content to the returned file path.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGetLogsDir(unittest.TestCase):
    """Tests for get_logs_dir()."""

    def test_get_logs_dir_creates_directory(self):
        """get_logs_dir() must create ~/.chat-workflow/logs/ and return the path."""
        from chat_workflow.session_logging import get_logs_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_home = Path(tmpdir)
            with patch("pathlib.Path.home", return_value=fake_home):
                result = get_logs_dir()

            expected = fake_home / ".chat-workflow" / "logs"
            self.assertEqual(result, expected)
            self.assertTrue(result.exists(), "The logs directory should exist")
            self.assertTrue(result.is_dir(), "The logs directory should be a directory")


class TestLogSession(unittest.TestCase):
    """Tests for log_session()."""

    def _sample_messages(self):
        return [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def _sample_criteria(self):
        return {
            "budget": {"weight": 8.0, "description": "Budget constraint"},
            "quality": {"weight": 7.0, "description": "Quality level"},
        }

    def test_log_session_writes_file(self):
        """log_session() must create a file at the returned path."""
        from chat_workflow.session_logging import log_session

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_logs_dir = Path(tmpdir)
            with patch(
                "chat_workflow.session_logging.get_logs_dir", return_value=fake_logs_dir
            ):
                result_path = log_session(
                    messages=self._sample_messages(),
                    criteria=self._sample_criteria(),
                    success_judgement=True,
                    feedback_text=None,
                    model="test-model",
                    turn_count=3,
                    context="test context",
                )

            self.assertIsInstance(result_path, Path)
            self.assertTrue(result_path.exists(), "The log file should exist on disk")
            self.assertTrue(
                result_path.is_file(), "The log file should be a regular file"
            )
            self.assertEqual(
                result_path.parent, fake_logs_dir, "File should be in the logs dir"
            )
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
        from chat_workflow.session_logging import log_session

        messages = self._sample_messages()
        criteria = self._sample_criteria()

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_logs_dir = Path(tmpdir)
            with patch(
                "chat_workflow.session_logging.get_logs_dir", return_value=fake_logs_dir
            ):
                result_path = log_session(
                    messages=messages,
                    criteria=criteria,
                    success_judgement=True,
                    feedback_text="Great session",
                    model="gpt-4",
                    turn_count=5,
                    context="testing content",
                )

            raw = result_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            self.assertEqual(data["model"], "gpt-4")
            self.assertEqual(data["turn_count"], 5)
            self.assertEqual(data["context"], "testing content")
            self.assertEqual(data["messages"], messages)
            self.assertEqual(data["criteria"], criteria)
            self.assertIn("timestamp", data)
            self.assertIn("user_feedback", data)
            self.assertEqual(data["user_feedback"]["success_judgement"], True)
            self.assertEqual(data["user_feedback"]["feedback_text"], "Great session")

    def test_log_session_none_criteria(self):
        """log_session() must handle criteria=None without error."""
        from chat_workflow.session_logging import log_session

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_logs_dir = Path(tmpdir)
            with patch(
                "chat_workflow.session_logging.get_logs_dir", return_value=fake_logs_dir
            ):
                result_path = log_session(
                    messages=self._sample_messages(),
                    criteria=None,
                    success_judgement=False,
                    feedback_text=None,
                    model="test-model",
                    turn_count=1,
                    context="none criteria test",
                )

            raw = result_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            self.assertIsNone(data["criteria"])
            self.assertEqual(data["model"], "test-model")
            self.assertEqual(data["turn_count"], 1)
            self.assertEqual(data["context"], "none criteria test")

    def test_log_session_with_feedback(self):
        """log_session() must include feedback_text in the output when provided."""
        from chat_workflow.session_logging import log_session

        feedback = "The model needs improvement on budget estimation."

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_logs_dir = Path(tmpdir)
            with patch(
                "chat_workflow.session_logging.get_logs_dir", return_value=fake_logs_dir
            ):
                result_path = log_session(
                    messages=self._sample_messages(),
                    criteria=self._sample_criteria(),
                    success_judgement=False,
                    feedback_text=feedback,
                    model="claude-3",
                    turn_count=7,
                    context="feedback test",
                )

            raw = result_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            self.assertEqual(data["user_feedback"]["success_judgement"], False)
            self.assertEqual(data["user_feedback"]["feedback_text"], feedback)
            self.assertEqual(data["model"], "claude-3")
            self.assertEqual(data["turn_count"], 7)
            self.assertEqual(data["context"], "feedback test")

    def test_log_session_empty_messages(self):
        """log_session() must handle an empty messages list."""
        from chat_workflow.session_logging import log_session

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_logs_dir = Path(tmpdir)
            with patch(
                "chat_workflow.session_logging.get_logs_dir", return_value=fake_logs_dir
            ):
                result_path = log_session(
                    messages=[],
                    criteria=None,
                    success_judgement=True,
                    feedback_text=None,
                    model="empty-test",
                    turn_count=0,
                    context="empty messages",
                )

            raw = result_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            self.assertEqual(data["messages"], [])
            self.assertEqual(data["turn_count"], 0)
            self.assertEqual(data["model"], "empty-test")

    def test_log_session_timestamp_format(self):
        """The timestamp field must be a valid ISO-8601 string."""
        from datetime import datetime

        from chat_workflow.session_logging import log_session

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_logs_dir = Path(tmpdir)
            with patch(
                "chat_workflow.session_logging.get_logs_dir", return_value=fake_logs_dir
            ):
                result_path = log_session(
                    messages=self._sample_messages(),
                    criteria=None,
                    success_judgement=True,
                    feedback_text=None,
                    model="ts-test",
                    turn_count=2,
                    context="timestamp check",
                )

            raw = result_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            parsed = datetime.fromisoformat(data["timestamp"])
            self.assertIsInstance(parsed, datetime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
