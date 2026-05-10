"""Workflow execution runner and error handling."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from chat_workflow import ConversationFlowState, ConversationIO, ConversationTools
from chat_workflow.config import Config
from chat_workflow.exceptions import (
    APIKeyError,
    ChatWorkflowError,
    ConfigFileError,
    ConfigurationError,
    ConversationFailedError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
    TurnLimitExceededError,
    ValidationError,
)
from chat_workflow.session_logging import log_session


class TyperConversationIO(ConversationIO):
    def echo(self, message: str) -> None:
        typer.echo(message)

    def prompt(self, label: str) -> str:
        return typer.prompt(label)


def handle_error(error: Exception) -> None:
    """Centralized error handler for CLI workflow execution."""
    if isinstance(error, ConfigFileError):
        typer.secho(
            f"\nConfiguration file error: {error.message}",
            err=True,
            fg=typer.colors.RED,
        )
    elif isinstance(error, ConfigurationError):
        typer.secho(f"\nConfiguration error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, APIKeyError):
        typer.secho(f"\nAPI key error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ProviderNotSupportedError):
        typer.secho(f"\nProvider error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ProviderNotFoundError):
        typer.secho(f"\nProvider not found: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, TurnLimitExceededError):
        typer.secho(f"\n{error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ConversationFailedError):
        typer.secho(f"\nConversation failed: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ValidationError):
        typer.secho(f"\nValidation error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ChatWorkflowError):
        typer.secho(f"\nError: {error.message}", err=True, fg=typer.colors.RED)
    else:
        typer.secho(f"\nUnexpected error: {str(error)[:200]}", err=True, fg=typer.colors.RED)
        message = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        typer.secho(f"\n{message}")
    raise typer.Exit(1)


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
        io = TyperConversationIO()
        flow_state = ConversationFlowState()
        config = Config(self.config_path)
        tools = ConversationTools(io=io, state=flow_state, config=config)

        def _log_and_exit(result_dict, default_success=True, feedback=None):
            judgement = typer.confirm("\nWas this experience successful?", default=default_success)
            if not judgement:
                feedback = typer.prompt("What went wrong? (optional)", default="")
                if feedback == "":
                    feedback = None
            try:
                path = log_session(
                    messages=flow_state.messages,
                    criteria=result_dict,
                    success_judgement=judgement,
                    feedback_text=feedback,
                    model=flow_state.model,
                    turn_count=flow_state.turn_count,
                    context=user_params.get("context", ""),
                )
                typer.echo(f"\nSession logged to: {path}")
            except Exception as log_err:
                typer.secho(f"\nFailed to log session: {log_err}", fg=typer.colors.YELLOW)

        try:
            user_params["tools"] = tools
            result = func(**user_params)
            _log_and_exit(result_dict=result.model_dump(), default_success=True)
        except KeyboardInterrupt:
            typer.echo("\n\nConversation cancelled.")
            raise typer.Exit(0) from None
        except ConversationFailedError as e:
            typer.secho(f"Conversation failed: {e.message}", err=True, fg=typer.colors.RED)
            _log_and_exit(result_dict=None, default_success=False)
            raise typer.Exit(1) from None
        except TurnLimitExceededError as e:
            typer.secho(f"{e.message}", err=True, fg=typer.colors.RED)
            _log_and_exit(result_dict=None, default_success=False, feedback="Turn limit reached")
            raise typer.Exit(1) from None
        except Exception as e:
            handle_error(e)
