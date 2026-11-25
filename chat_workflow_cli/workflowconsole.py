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

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (Path(__file__).parent.parent / "config.json")
        self._session_log = SessionLog()
        self._context = ""

    def run(
        self,
        func: Callable[..., Any],
        # lucidlint: ignore record-shape the step's own kwargs, discovered from its signature
        user_params: dict[str, Any],
    ) -> None:
        """Run a workflow function with framework plumbing."""
        typer.echo("Starting conversation... (Ctrl+C to quit)")
        io = TyperUserIO()
        config = Config(self.config_path)
        session = Session(io=io, state=self._session_log, config=config)
        user_params.pop("session", None)
        self._context = user_params.get("context", "")

        try:
            step = _BoundStep(
                functools.partial(func, **user_params),
                getattr(func, "_chat", True),
            )
            result = SessionComponent().converse(step, session=session)
            if result is not None:
                self._log_and_exit(result_dict=result.model_dump(), default_success=True)
            else:
                typer.echo("\nDone.")
        except KeyboardInterrupt:
            typer.echo("\n\nConversation cancelled.")
            raise typer.Exit(0) from None
        except AtomicWorkflowFailedError as e:
            typer.secho(f"Atomic workflow failed: {e.message}", err=True, fg=typer.colors.RED)
            self._log_and_exit(result_dict=None, default_success=False)
            raise typer.Exit(1) from None
        except TurnLimitExceededError as e:
            typer.secho(f"{e.message}", err=True, fg=typer.colors.RED)
            self._log_and_exit(
                result_dict=None, default_success=False, feedback="Turn limit reached"
            )
            raise typer.Exit(1) from None
        # lucidlint: ignore swallow the CLI's top boundary: save the crash, then render it and exit through handle_error
        except Exception as e:
            self._save_crash(e)
            handle_error(e)

    def _log_and_exit(
        self,
        result_dict,
        default_success: bool = True,
        feedback: str | None = None,
    ) -> None:
        judgement = typer.confirm("\nWas this experience successful?", default=default_success)
        if not judgement:
            feedback = typer.prompt("What went wrong? (optional)", default="") or None
        try:
            path = self._write_session(result_dict, judgement, feedback)
            typer.echo(f"\nSession logged to: {path}")
        # lucidlint: ignore swallow the report is a boundary: logging must not fail a done run
        except Exception as log_err:
            typer.secho(f"\nFailed to log session: {log_err}", fg=typer.colors.YELLOW)

    def _save_crash(self, error: Exception) -> None:
        """Save the transcript and companion traceback without prompting."""
        try:
            path = self._write_session(None, success=False, feedback="Unhandled exception")
            typer.echo(f"\nSession logged to: {path}")
            tb_path = path.with_name(path.stem + "-exception.txt")
            tb_path.write_text(
                "=== EXCEPTION DETAILS ===\n"
                + "".join(traceback.format_exception(type(error), error, error.__traceback__))
                + "\n=== END EXCEPTION DETAILS ===\n"
            )
            typer.echo(f"Exception details written to: {tb_path}")
        # lucidlint: ignore swallow the crash report is a boundary: logging must not mask the error
        except Exception as log_err:
            typer.secho(f"\nFailed to log session: {log_err}", fg=typer.colors.YELLOW)

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
        return log_session(record)
