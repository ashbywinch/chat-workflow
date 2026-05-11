#!/usr/bin/env python3
"""CLI entrypoint with automatic workflow discovery."""

import importlib
import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from .workflow_runner import WorkflowRunner

# Parameters injected by the framework, not exposed as CLI options
_INTERNAL_PARAMS = frozenset({"session", "debug"})


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


def _build_workflow_sub_app(workflow_name: str) -> typer.Typer | None:
    """Build a Typer sub-app for a single workflow module.

    Imports the module via ``workflows.{workflow_name}``, discovers
    ``@composite_workflow``-decorated functions, and registers each as a subcommand.

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
            runner = WorkflowRunner()
            runner.run(_func, workflow_params)

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
