"""Workflow execution runner and error handling."""

from __future__ import annotations

import traceback

import typer

from chat_workflow.exceptions import (
    APIKeyError,
    AtomicWorkflowFailedError,
    ChatWorkflowError,
    ConfigFileError,
    ConfigurationError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
    TurnLimitExceededError,
    ValidationError,
)
from chat_workflow.session import UserIO


class TyperUserIO(UserIO):
    def echo(self, message: str) -> None:
        typer.echo(message)

    def prompt(self, label: str, **kwargs: object) -> str:
        return typer.prompt(label, **kwargs)


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
    elif isinstance(error, AtomicWorkflowFailedError):
        typer.secho(f"\nAtomic workflow failed: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ValidationError):
        typer.secho(f"\nValidation error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ChatWorkflowError):
        typer.secho(f"\nError: {error.message}", err=True, fg=typer.colors.RED)
    else:
        typer.secho(f"\nUnexpected error: {str(error)[:200]}", err=True, fg=typer.colors.RED)
        message = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        typer.secho(f"\n{message}")
    raise typer.Exit(1)
