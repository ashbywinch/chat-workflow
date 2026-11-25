import functools
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from chat_workflow.config import Config
from chat_workflow.exceptions import AtomicWorkflowFailedError, TurnLimitExceededError
from chat_workflow.session import Session
from chat_workflow.session_component import SessionComponent
from chat_workflow.session_log import SessionLog
from chat_workflow.session_logging import SessionRecord, log_session

from .runner import TyperUserIO, handle_error

__all__ = ["WorkflowConsole"]


class _BoundStep:
    """The workflow function bound to its CLI arguments, carrying its chat flag."""

    def __init__(self, call: Callable[..., Any], chat: bool):
        self._call = call
        self._chat = chat

    def __call__(self, *, session: Session) -> Any:
        return self._call(session=session)


class WorkflowConsole:
    """Hosts one workflow conversation: IO, session state, config and logging."""

    # lucidlint: ignore long-param-list the collaborators are independent injection seams
    def __init__(
        self,
        config_path: Path | None = None,
        *,
        config_factory: Callable[[Path], Any] = Config,
        session_factory: Callable[..., Any] = Session,
        io_factory: Callable[[], Any] = TyperUserIO,
        component_factory: Callable[[], Any] = SessionComponent,
        logger: Callable[[SessionRecord], Path] = log_session,
        error_handler: Callable[[Exception], None] = handle_error,
        console: Any = typer,
    ):
        self.config_path = config_path or (Path(__file__).parent.parent / "config.json")
        self._session_log = SessionLog()
        self._context = ""
        # Collaborators are injectable so a test can supply object fakes
        # instead of monkeypatching module globals.
        self._config_factory = config_factory
        self._session_factory = session_factory
        self._io_factory = io_factory
        self._component_factory = component_factory  # lucidlint: ignore duplicate-block each line stores a collaborator
        self._logger = logger
        self._error_handler = error_handler
        self._console = console

    def run(
        self,
        func: Callable[..., Any],
        # lucidlint: ignore record-shape the step's own kwargs, discovered from its signature
        user_params: dict[str, Any],
    ) -> None:
        """Run a workflow function with framework plumbing."""
        self._console.echo("Starting conversation... (Ctrl+C to quit)")
        io = self._io_factory()
        config = self._config_factory(self.config_path)
        session = self._session_factory(io=io, state=self._session_log, config=config)
        user_params.pop("session", None)
        self._context = user_params.get("context", "")

        try:
            step = _BoundStep(
                functools.partial(func, **user_params),
                getattr(func, "_chat", True),
            )
            result = self._component_factory().converse(step, session=session)
            if result is not None:
                self._log_and_exit(result_dict=result.model_dump(), default_success=True)
            else:
                self._console.echo("\nDone.")
        except KeyboardInterrupt:
            self._console.echo("\n\nConversation cancelled.")
            raise self._console.Exit(0) from None
        except AtomicWorkflowFailedError as e:
            self._console.secho(
                f"Atomic workflow failed: {e.message}", err=True, fg=self._console.colors.RED
            )
            self._log_and_exit(result_dict=None, default_success=False)
            raise self._console.Exit(1) from None
        except TurnLimitExceededError as e:
            self._console.secho(f"{e.message}", err=True, fg=self._console.colors.RED)
            self._log_and_exit(
                result_dict=None, default_success=False, feedback="Turn limit reached"
            )
            raise self._console.Exit(1) from None
        # lucidlint: ignore swallow the CLI's top boundary: save the crash, then render it and exit through handle_error
        except Exception as e:
            self._save_crash(e)
            self._error_handler(e)

    def _log_and_exit(
        self,
        result_dict,
        default_success: bool = True,
        feedback: str | None = None,
    ) -> None:
        judgement = self._console.confirm(
            "\nWas this experience successful?", default=default_success
        )
        if not judgement:
            feedback = self._console.prompt("What went wrong? (optional)", default="") or None
        try:
            path = self._write_session(result_dict, judgement, feedback)
            self._console.echo(f"\nSession logged to: {path}")
        # lucidlint: ignore swallow the report is a boundary: logging must not fail a done run
        except Exception as log_err:
            self._console.secho(
                f"\nFailed to log session: {log_err}", fg=self._console.colors.YELLOW
            )

    def _save_crash(self, error: Exception) -> None:
        """Save the transcript and companion traceback without prompting."""
        try:
            path = self._write_session(None, success=False, feedback="Unhandled exception")
            self._console.echo(f"\nSession logged to: {path}")
            tb_path = path.with_name(path.stem + "-exception.txt")
            tb_path.write_text(
                "=== EXCEPTION DETAILS ===\n"
                + "".join(traceback.format_exception(type(error), error, error.__traceback__))
                + "\n=== END EXCEPTION DETAILS ===\n"
            )
            self._console.echo(f"Exception details written to: {tb_path}")
        # lucidlint: ignore swallow the crash report is a boundary: logging must not mask the error
        except Exception as log_err:
            self._console.secho(
                f"\nFailed to log session: {log_err}", fg=self._console.colors.YELLOW
            )

    def _report_log_failure(self, error: Exception) -> None:
        self._console.secho(
            f"\nFailed to log session: {error}", fg=self._console.colors.YELLOW
        )

    def _write_session(self, criteria, success: bool, feedback: str | None) -> Path:
        record = SessionRecord(
            messages=self._session_log.messages,
            criteria=criteria,
            success_judgement=success,
            feedback_text=feedback,
            model=self._session_log.model,
            turn_count=self._session_log.turn_count,
            context=self._context,
        )
        return self._logger(record)
