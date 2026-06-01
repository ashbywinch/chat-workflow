#!/usr/bin/env python3
"""Tests for handle_error() — traceback sanitization.

handle_error() must NEVER print Python tracebacks to end users.
ChatWorkflowError subclasses must still show their friendly messages.
The generic else branch must keep the sanitized str(error)[:200] display.
"""

import unittest
from unittest.mock import patch

from typer import Exit as TyperExit

from chat_workflow.exceptions import (
    APIKeyError,
    ChatWorkflowError,
    ConfigFileError,
    ValidationError,
)


class TestHandleError(unittest.TestCase):
    """handle_error must never leak tracebacks to users."""

    def setUp(self):
        self.patcher = patch("chat_workflow_cli.runner.typer.secho")
        self.mock_secho = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _call_handle_error(self, error: Exception) -> None:
        """Call handle_error and assert it raises typer.Exit(1)."""
        from chat_workflow_cli.runner import handle_error

        with self.assertRaises(TyperExit) as ctx:
            handle_error(error)
        self.assertEqual(ctx.exception.exit_code, 1)

    # ── Generic Exception (else branch) ──────────────────────────────

    def test_generic_exception_shows_sanitized_message(self):
        """Generic Exception shows 'Unexpected error: {str(error)[:200]}'."""
        error = ValueError("something broke deeply")
        self._call_handle_error(error)

        self.mock_secho.assert_called_once()
        (msg,), kwargs = self.mock_secho.call_args
        self.assertIn("Unexpected error: something broke deeply", msg)
        self.assertTrue(kwargs.get("err", False))

    def test_generic_exception_no_traceback_in_output(self):
        """Generic Exception must not contain traceback content in output."""
        error = RuntimeError("runtime issue")
        self._call_handle_error(error)

        (msg,), _ = self.mock_secho.call_args
        self.assertNotIn("Traceback", msg)
        self.assertNotIn('File "', msg)

    def test_generic_exception_message_truncated_to_200_chars(self):
        """str(error) is truncated to 200 characters."""
        long_msg = "x" * 500
        error = ValueError(long_msg)
        self._call_handle_error(error)

        (msg,), _ = self.mock_secho.call_args
        # Should contain only the first 200 chars of the message
        self.assertIn(long_msg[:200], msg)
        self.assertNotIn(long_msg[200:], msg)

    # ── ChatWorkflowError subclasses ─────────────────────────────────

    def test_chat_workflow_error_shows_error_prefix(self):
        """Plain ChatWorkflowError shows 'Error:' prefix."""
        error = ChatWorkflowError("generic workflow issue")
        self._call_handle_error(error)

        self.mock_secho.assert_called_once()
        (msg,), _ = self.mock_secho.call_args
        self.assertIn("Error: generic workflow issue", msg)

    def test_api_key_error_shows_friendly_message(self):
        """APIKeyError shows 'Configuration error:' prefix (inherits ConfigurationError)."""
        error = APIKeyError("Missing API key")
        self._call_handle_error(error)

        self.mock_secho.assert_called_once()
        (msg,), _ = self.mock_secho.call_args
        # APIKeyError inherits from ConfigurationError, so the
        # ConfigurationError branch catches it first.
        self.assertIn("Configuration error: Missing API key", msg)

    def test_api_key_error_no_traceback(self):
        """APIKeyError output must not contain traceback content."""
        error = APIKeyError("invalid key")
        self._call_handle_error(error)

        (msg,), _ = self.mock_secho.call_args
        self.assertNotIn("Traceback", msg)
        self.assertNotIn('File "', msg)

    def test_config_file_error_shows_specific_prefix(self):
        """ConfigFileError shows 'Configuration file error:' prefix."""
        error = ConfigFileError("config.json not found")
        self._call_handle_error(error)

        (msg,), _ = self.mock_secho.call_args
        self.assertIn("Configuration file error: config.json not found", msg)

    def test_validation_error_shows_validation_prefix(self):
        """ValidationError shows 'Validation error:' prefix."""
        error = ValidationError("invalid data")
        self._call_handle_error(error)

        (msg,), _ = self.mock_secho.call_args
        self.assertIn("Validation error: invalid data", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
