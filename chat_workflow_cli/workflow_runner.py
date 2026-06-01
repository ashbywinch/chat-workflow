import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from chat_workflow.config import Config
from chat_workflow.exceptions import AtomicWorkflowFailedError, TurnLimitExceededError
from chat_workflow.session import Session
from chat_workflow.session_log import SessionLog
from chat_workflow.session_logging import log_session

from .runner import TyperUserIO, handle_error


class WorkflowRunner:
    """Runs a workflow function with framework plumbing (IO, state, config, logging)."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (Path(__file__).parent.parent / "config.json")

    def run(
        self,
        func: Callable[..., Any],
        user_params: dict[str, Any],
    ) -> None:
        """Run a workflow function with framework plumbing."""
        typer.echo("Starting conversation... (Ctrl+C to quit)")
        io = TyperUserIO()
        session_log = SessionLog()
        config = Config(self.config_path)
        session = Session(io=io, state=session_log, config=config)

        def _log_and_exit(result_dict, default_success=True, feedback=None):
            judgement = typer.confirm("\nWas this experience successful?", default=default_success)
            if not judgement:
                feedback = typer.prompt("What went wrong? (optional)", default="")
                if feedback == "":
                    feedback = None
            try:
                path = log_session(
                    messages=session_log.messages,
                    criteria=result_dict,
                    success_judgement=judgement,
                    feedback_text=feedback,
                    model=session_log.model,
                    turn_count=session_log.turn_count,
                    context=user_params.get("context", ""),
                )
                typer.echo(f"\nSession logged to: {path}")
            except Exception as log_err:
                typer.secho(f"\nFailed to log session: {log_err}", fg=typer.colors.YELLOW)

        try:
            user_params["session"] = session
            result = func(**user_params)
            if result is not None:
                _log_and_exit(result_dict=result.model_dump(), default_success=True)
            else:
                typer.echo("\nDone.")
        except KeyboardInterrupt:
            typer.echo("\n\nConversation cancelled.")
            raise typer.Exit(0) from None
        except AtomicWorkflowFailedError as e:
            typer.secho(f"Atomic workflow failed: {e.message}", err=True, fg=typer.colors.RED)
            _log_and_exit(result_dict=None, default_success=False)
            raise typer.Exit(1) from None
        except TurnLimitExceededError as e:
            typer.secho(f"{e.message}", err=True, fg=typer.colors.RED)
            _log_and_exit(result_dict=None, default_success=False, feedback="Turn limit reached")
            raise typer.Exit(1) from None
        except Exception as e:
            # Save session log automatically on unhandled exceptions (no feedback prompt)
            try:
                path = log_session(
                    messages=session_log.messages,
                    criteria=None,
                    success_judgement=False,
                    feedback_text="Unhandled exception",
                    model=session_log.model,
                    turn_count=session_log.turn_count,
                    context=user_params.get("context", ""),
                )
                typer.echo(f"\nSession logged to: {path}")
                tb_path = path.with_name(path.stem + "-exception.txt")
                tb_path.write_text(
                    "=== EXCEPTION DETAILS ===\n"
                    + "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    + "\n=== END EXCEPTION DETAILS ===\n"
                )
                typer.echo(f"Exception details written to: {tb_path}")
            except Exception as log_err:
                typer.secho(f"\nFailed to log session: {log_err}", fg=typer.colors.YELLOW)
            handle_error(e)
