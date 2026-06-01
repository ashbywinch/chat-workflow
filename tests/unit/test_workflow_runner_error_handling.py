#!/usr/bin/env python3
"""Tests for generic exception handling in WorkflowRunner.run()."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_workflow_cli.workflow_runner import WorkflowRunner


class TestWorkflowRunnerErrorHandling(unittest.TestCase):
    """Verifies the generic Exception handler saves logs + companion file before handle_error."""

    def _failing_func(self, **kwargs):
        """A workflow function that raises an unhandled exception."""
        raise ValueError("test crash: something went wrong")

    def _get_runner(self, config_path: Path) -> WorkflowRunner:
        """Create a WorkflowRunner with the given config path."""
        return WorkflowRunner(config_path=config_path)

    # ----------------------------------------------------------------
    # Test 1: Companion -exception.txt file is created with traceback
    # ----------------------------------------------------------------
    @patch("chat_workflow_cli.workflow_runner.Config")
    @patch("chat_workflow_cli.workflow_runner.Session")
    @patch("chat_workflow_cli.workflow_runner.handle_error")
    @patch("chat_workflow_cli.workflow_runner.typer")
    def test_companion_exception_file_created(
        self,
        mock_typer,
        mock_handle_error,
        mock_session_class,
        mock_config_class,
    ):
        """Exception handler must write companion -exception.txt with full traceback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "session_test.json"
            # Pre-create the log file so the companion path derivation works
            log_path.write_text("{}")

            with patch(
                "chat_workflow_cli.workflow_runner.log_session",
                return_value=log_path,
            ) as mock_log_session:
                runner = self._get_runner(config_path=Path(tmpdir) / "config.json")
                runner.run(func=self._failing_func, user_params={})

            # Companion file must exist
            expected_tb_path = log_path.with_name(log_path.stem + "-exception.txt")
            self.assertTrue(
                expected_tb_path.exists(),
                f"Companion exception file not found at {expected_tb_path}",
            )

            # Content must contain markers and traceback
            content = expected_tb_path.read_text()
            self.assertIn("=== EXCEPTION DETAILS ===", content)
            self.assertIn("=== END EXCEPTION DETAILS ===", content)
            self.assertIn("ValueError", content)
            self.assertIn("test crash: something went wrong", content)

            # log_session must have been called with crash-specific args
            mock_log_session.assert_called_once()
            _, kwargs = mock_log_session.call_args
            self.assertFalse(kwargs["success_judgement"])
            self.assertEqual(kwargs["feedback_text"], "Unhandled exception")
            self.assertIsNone(kwargs["criteria"])

    # ----------------------------------------------------------------
    # Test 2: handle_error() IS called after companion file save
    # ----------------------------------------------------------------
    @patch("chat_workflow_cli.workflow_runner.Config")
    @patch("chat_workflow_cli.workflow_runner.Session")
    @patch("chat_workflow_cli.workflow_runner.log_session")
    @patch("chat_workflow_cli.workflow_runner.typer")
    def test_handle_error_called_after_save(
        self,
        mock_typer,
        mock_log_session,
        mock_session_class,
        mock_config_class,
    ):
        """handle_error() must be invoked after the session and companion file are saved."""
        # Make log_session return a real-ish path so .with_name().write_text() doesn't crash
        import tempfile

        mock_log_path = Path(tempfile.gettempdir()) / "session_test_handle.json"

        with patch(
            "chat_workflow_cli.workflow_runner.handle_error",
        ) as mock_handle_error:
            mock_log_session.return_value = mock_log_path

            runner = self._get_runner(config_path=Path("/nonexistent/config.json"))
            runner.run(func=self._failing_func, user_params={})

            mock_handle_error.assert_called_once()
            args, _ = mock_handle_error.call_args
            self.assertIsInstance(args[0], ValueError)
            self.assertEqual(str(args[0]), "test crash: something went wrong")

    # ----------------------------------------------------------------
    # Test 3: typer.confirm is NOT called on the crash path
    # ----------------------------------------------------------------
    @patch("chat_workflow_cli.workflow_runner.Config")
    @patch("chat_workflow_cli.workflow_runner.Session")
    @patch("chat_workflow_cli.workflow_runner.log_session")
    @patch("chat_workflow_cli.workflow_runner.handle_error")
    def test_no_feedback_prompt_on_crash(
        self,
        mock_handle_error,
        mock_log_session,
        mock_session_class,
        mock_config_class,
    ):
        """typer.confirm must NOT be called when an unhandled exception occurs."""
        mock_log_path = Path("/tmp/session_test_noprompt.json")

        with patch(
            "chat_workflow_cli.workflow_runner.typer.confirm",
        ) as mock_confirm:
            mock_log_session.return_value = mock_log_path

            runner = self._get_runner(config_path=Path("/nonexistent/config.json"))
            runner.run(func=self._failing_func, user_params={})

            # confirm is used in _log_and_exit (success/specific-error path)
            # but must NEVER be called on the generic Exception path
            mock_confirm.assert_not_called()
