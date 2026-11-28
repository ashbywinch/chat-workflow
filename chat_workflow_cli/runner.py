"""Workflow execution runner and error handling."""

from __future__ import annotations

import traceback
from typing import override

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

__all__ = ["TyperUserIO", "handle_error"]


class TyperUserIO(UserIO):
    @override
    def echo(self, message: str) -> None:
        typer.echo(message)

    @staticmethod
    @override
    def prompt(label: str) -> str:
        return typer.prompt(label)


def _fail(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)


def handle_error(error: Exception) -> None:
    """Render the error in the CLI's voice, then exit with a failure status.

    One message per known exception kind; anything else gets the traceback.
    """
    templates = {
        ConfigFileError: "\nConfiguration file error: {message}",
        ConfigurationError: "\nConfiguration error: {message}",
        APIKeyError: "\nAPI key error: {message}",
        ProviderNotSupportedError: "\nProvider error: {message}",
        ProviderNotFoundError: "\nProvider not found: {message}",
        TurnLimitExceededError: "\n{message}",
        AtomicWorkflowFailedError: "\nAtomic workflow failed: {message}",
        ValidationError: "\nValidation error: {message}",
        ChatWorkflowError: "\nError: {message}",
    }
    for error_type, template in templates.items():
        if isinstance(error, error_type):
            _fail(template.format(message=error.message))
            break
    else:
        _fail(f"\nUnexpected error: {str(error)[:200]}")
        message = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        _fail(f"\n{message}")
    raise typer.Exit(1)
