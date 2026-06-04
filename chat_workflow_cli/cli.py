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
        if inspect.isfunction(obj) and getattr(obj, "_is_workflow", False) and not name.startswith("_"):
            functions[name] = obj
    return functions


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def _register_command(app: typer.Typer, func_name: str, func: Callable[..., Any]) -> None:
    """Register a workflow function as a command on a Typer app."""
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

    app.command(name=_snake_to_kebab(func_name))(run_cmd)


def _discover_class_workflow_methods(module) -> dict[str, dict[str, Callable[..., Any]]]:
    """Discover {class_name: {method_name: callable}} from exported classes with _is_workflow methods."""
    exported = set(getattr(module, "__all__", []))
    classes: dict[str, dict[str, Callable[..., Any]]] = {}
    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj) or name not in exported:
            continue
        methods: dict[str, Callable[..., Any]] = {}
        for mname, mobj in inspect.getmembers(obj):
            if not getattr(mobj, "_is_workflow", False) or mname.startswith("_"):
                continue
            methods[mname] = mobj
        if methods:
            classes[name] = methods
    return classes


def _build_workflow_sub_app(workflow_name: str) -> typer.Typer | None:
    """Build a Typer sub-app for a single workflow module.

    Imports the module via ``workflows.{workflow_name}``, discovers
    ``@composite_workflow``-decorated module-level functions AND
    classmethods, and registers each.

    Module-level functions become flat subcommand (e.g. ``generate-reviewed-criteria``).
    Class-level methods become a sub-app with the class name
    (e.g. ``Workflow create``).

    Returns *None* when the module cannot be imported or contains no
    workflow functions — the caller should skip it silently.
    """
    module_name = f"workflows.{workflow_name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    # Module-level workflow functions (e.g. generate_reviewed_criteria)
    functions = discover_workflow_functions(module)

    # Class-level workflow methods (e.g. Workflow.create)
    class_methods = _discover_class_workflow_methods(module)

    if not functions and not class_methods:
        return None

    sub_app = typer.Typer(help=f"Commands for {workflow_name} workflow")

    # Register module-level functions as flat commands
    for func_name, func in functions.items():
        _register_command(sub_app, func_name, func)

    # Register class-level methods as class sub-apps
    for class_name, methods in class_methods.items():
        class_app = typer.Typer(help=f"Commands for {class_name}")
        for mname, mobj in methods.items():
            _register_command(class_app, mname, mobj)
        sub_app.add_typer(class_app, name=_snake_to_kebab(class_name))

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
