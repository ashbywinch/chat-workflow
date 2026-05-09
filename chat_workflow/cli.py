#!/usr/bin/env python3
"""CLI entrypoint with automatic workflow discovery."""

import importlib
import inspect
import traceback
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from chat_workflow import (
    ConversationFlowState,
    ConversationIO,
    ConversationTools,
)
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

# Parameters injected by the framework, not exposed as CLI options
_INTERNAL_PARAMS = frozenset({"tools", "io", "state", "debug"})


class TyperConversationIO(ConversationIO):
    def echo(self, message: str) -> None:
        typer.echo(message)

    def prompt(self, label: str) -> str:
        return typer.prompt(label)


def handle_error(error: Exception):
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


def discover_workflows() -> dict[str, Path]:
    workflows_dir = Path(__file__).parent.parent / "workflows"
    workflows = {}
    if not workflows_dir.exists():
        return workflows
    for item in workflows_dir.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            workflows[item.name] = item
    return workflows


def discover_workflow_functions(module) -> dict[str, Callable[..., Any]]:
    functions = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and getattr(obj, "_is_workflow", False):
            functions[name] = obj
    return functions


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def _execute_workflow(
    func: Callable[..., Any],
    user_params: dict[str, Any],
):
    """Run a workflow function with framework plumbing."""
    typer.echo("Starting conversation... (Ctrl+C to quit)")
    io = TyperConversationIO()
    flow_state = ConversationFlowState()
    config = Config(Path(__file__).parent.parent / "config.json")
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


def _build_workflow_sub_app(workflow_name: str) -> typer.Typer | None:
    """Build a Typer sub-app for a single workflow module.

    Imports the module via ``workflows.{workflow_name}``, discovers
    ``@workflow``-decorated functions, and registers each as a subcommand.

    Returns *None* when the module cannot be imported or contains no
    workflow functions — the caller should skip it silently.
    """
    module_name = f"workflows.{workflow_name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    functions = discover_workflow_functions(module)
    if not functions:
        return None

    sub_app = typer.Typer(help=f"Commands for {workflow_name} workflow")

    for func_name, func in functions.items():
        sig = inspect.signature(func)
        try:
            type_hints = typing.get_type_hints(func, include_extras=True)
        except Exception:
            type_hints = {}

        user_params = []
        for param_name, param in sig.parameters.items():
            if param_name not in _INTERNAL_PARAMS:
                user_params.append(param)

        sig_params = []
        for param in user_params:
            annotation = type_hints.get(param.name, str)
            default = param.default if param.default is not inspect.Parameter.empty else None
            sig_params.append(
                inspect.Parameter(
                    param.name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )

        def run_cmd(*args, _func=func, **workflow_params):
            _execute_workflow(_func, workflow_params)

        run_cmd.__signature__ = inspect.Signature(
            parameters=sig_params,
            return_annotation=inspect.Parameter.empty,
        )

        sub_app.command(name=_snake_to_kebab(func_name))(run_cmd)

    return sub_app


def build_cli_app() -> typer.Typer:
    app = typer.Typer(help="Chat Workflow CLI - Generate structured data through LLM conversations")

    for workflow_name in discover_workflows():
        sub_app = _build_workflow_sub_app(workflow_name)
        if sub_app is None:
            continue
        kebab_name = _snake_to_kebab(workflow_name)
        app.add_typer(sub_app, name=kebab_name)

    return app


app = build_cli_app()

if __name__ == "__main__":
    app()
