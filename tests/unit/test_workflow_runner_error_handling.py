#!/usr/bin/env python3
"""Tests for generic exception handling in WorkflowConsole.run().

The console takes its collaborators by injection, so the crash path is
exercised with object fakes — no module globals are patched.
"""

import unittest
from pathlib import Path
from typing import override

from pyfakefs.fake_filesystem_unittest import TestCase

from chat_workflow.turn_budget import DEFAULT_SESSION_MAX_TURNS
from chat_workflow_cli.workflowconsole import WorkflowConsole
from tests.conftest import FakeConfig


class _FakeIO:
    def echo(self, message: str) -> None:
        pass

    def prompt(self, label: str) -> str:
        return ""


class _NoWrapComponent:
    """Run the step directly: the crash path, not the greeting, is the subject."""

    def converse(self, step, *, session):
        return step(session=session)


class _FakeExitError(Exception):
    pass


class _FakeColors:
    RED = "red"
    YELLOW = "yellow"


class _FakeConsole:
    Exit = _FakeExitError
    colors = _FakeColors

    def __init__(self) -> None:
        self.echoes: list[str] = []
        self.sechoes: list[str] = []
        self.confirms: list[str] = []
        self.prompts: list[str] = []

    def echo(self, message: str) -> None:
        self.echoes.append(message)

    def secho(self, message: str, **_kwargs) -> None:
        self.sechoes.append(message)

    def confirm(self, message: str, *, default: bool = False) -> bool:
        self.confirms.append(message)
        return default

    def prompt(self, message: str, *, default: str = "") -> str:
        self.prompts.append(message)
        return default


class _FakeLogger:
    """A session logger that records the record and returns a chosen path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records = []

    def __call__(self, record):
        self.records.append(record)
        return self.path


class _FakeErrorHandler:
    def __init__(self) -> None:
        self.errors: list[Exception] = []

    def __call__(self, error: Exception) -> None:
        self.errors.append(error)


class TestSessionTurnBudgetOption(unittest.TestCase):
    """The CLI option is the session guard, not a workflow parameter."""

    def _sessions(self):
        sessions = []

        def session_factory(**kwargs):
            sessions.append(kwargs)
            return object()

        console = WorkflowConsole(
            config_factory=lambda _path: FakeConfig(),
            io_factory=_FakeIO,
            session_factory=session_factory,
            component_factory=_NoWrapComponent,
            console=_FakeConsole(),
        )
        return console, sessions

    def test_max_turns_sets_the_session_budget(self):
        console, sessions = self._sessions()

        console.run(func=lambda **kwargs: None, user_params={"max_turns": 7})

        self.assertEqual(sessions[0]["budget"].max_turns, 7)

    def test_default_is_the_framework_guard(self):
        console, sessions = self._sessions()

        console.run(func=lambda **kwargs: None, user_params={})

        self.assertEqual(
            sessions[0]["budget"].max_turns, DEFAULT_SESSION_MAX_TURNS
        )


class TestWorkflowConsoleErrorHandling(TestCase):
    """Verifies the generic Exception handler saves logs + companion file before handling."""

    @override
    def setUp(self) -> None:
        self.setUpPyfakefs()

    def _failing_func(self, **kwargs):
        """A workflow function that raises an unhandled exception."""
        raise ValueError("test crash: something went wrong")

    def _console(
        self, log_path: Path, **overrides
    ) -> tuple[WorkflowConsole, _FakeLogger, _FakeErrorHandler, _FakeConsole]:
        """A console wired with object fakes and a fake log destination."""
        self.fs.create_dir(str(log_path.parent))
        log_path.write_text("{}")
        dependencies = {
            "config_factory": lambda _path: FakeConfig(),
            "io_factory": _FakeIO,
            "component_factory": _NoWrapComponent,
            "logger": _FakeLogger(log_path),
            "error_handler": _FakeErrorHandler(),
            "console": _FakeConsole(),
        }
        dependencies.update(overrides)
        console = WorkflowConsole(
            config_path=log_path.parent / "config.json",
            **dependencies,
        )
        return (
            console,
            dependencies["logger"],
            dependencies["error_handler"],
            dependencies["console"],
        )

    def test_companion_exception_file_created(self):
        """Exception handler must write companion -exception.txt with full traceback."""
        log_path = Path("/logs/session_test.json")
        console, logger, _handler, _terminal = self._console(log_path)

        console.run(func=self._failing_func, user_params={})

        expected_tb_path = log_path.with_name(log_path.stem + "-exception.txt")
        self.assertTrue(
            expected_tb_path.exists(),
            f"Companion exception file not found at {expected_tb_path}",
        )

        content = expected_tb_path.read_text()
        self.assertIn("=== EXCEPTION DETAILS ===", content)
        self.assertIn("=== END EXCEPTION DETAILS ===", content)
        self.assertIn("ValueError", content)
        self.assertIn("test crash: something went wrong", content)

        self.assertEqual(len(logger.records), 1)
        record = logger.records[0]
        self.assertFalse(record.success_judgement)
        self.assertEqual(record.feedback_text, "Unhandled exception")
        self.assertIsNone(record.criteria)

    def test_handle_error_called_after_save(self):
        """The error handler must run after the session and companion file are saved."""
        log_path = Path("/logs/session_test_handle.json")
        console, _logger, handler, _terminal = self._console(log_path)

        console.run(func=self._failing_func, user_params={})

        self.assertEqual(len(handler.errors), 1)
        self.assertIsInstance(handler.errors[0], ValueError)
        self.assertEqual(str(handler.errors[0]), "test crash: something went wrong")

    def test_no_feedback_prompt_on_crash(self):
        """confirm must NOT be called when an unhandled exception occurs."""
        log_path = Path("/logs/session_test_noprompt.json")
        console, _logger, _handler, terminal = self._console(log_path)

        console.run(func=self._failing_func, user_params={})

        self.assertEqual(terminal.confirms, [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
