"""Smoke tests for the CLI app — no API keys or network required."""

import unittest
from typer.testing import CliRunner

from chat_workflow.cli import app


class TestCliSmoke(unittest.TestCase):
    """Verify the CLI boots, parses commands, and renders help without crashing."""

    runner = CliRunner()

    def test_root_help_exits_zero(self):
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Chat Workflow CLI", result.output)

    def test_no_args_shows_help(self):
        """Typer groups show usage info when no subcommand is given."""
        result = self.runner.invoke(app)
        self.assertIn("Missing command", result.output)

    def test_workflow_subcommand_help(self):
        """Discovered workflows should appear in help and be invocable."""
        result = self.runner.invoke(app, ["evaluation-criteria", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Commands for", result.output)

    def test_workflow_function_help(self):
        """A specific workflow function should render its own help with known options."""
        result = self.runner.invoke(
            app, ["evaluation-criteria", "generate-reviewed-criteria", "--help"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--context", result.output)

    def test_invalid_workflow_name_errors(self):
        result = self.runner.invoke(app, ["nonexistent-workflow"])
        self.assertNotEqual(result.exit_code, 0)

    def test_invalid_command_errors(self):
        result = self.runner.invoke(
            app, ["evaluation-criteria", "nonexistent-command"]
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_invalid_option_errors(self):
        result = self.runner.invoke(
            app, ["evaluation-criteria", "generate-reviewed-criteria", "--bogus"]
        )
        self.assertNotEqual(result.exit_code, 0)