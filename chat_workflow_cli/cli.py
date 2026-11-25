#!/usr/bin/env python3
"""CLI entrypoint with automatic workflow discovery."""

import importlib
import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from .workflowconsole import WorkflowConsole

# Parameters injected by the framework, not exposed as CLI options
_INTERNAL_PARAMS = frozenset({"session", "debug"})


@dataclass(frozen=True)
class WorkflowModule:
    """One workflow package the CLI can expose."""

    name: str
    path: Path


@dataclass(frozen=True)
class WorkflowCommand:
    """One runnable workflow function or method."""

    name: str
    func: Callable[..., Any]


@dataclass(frozen=True)
class WorkflowClass:
    """An exported class whose public methods are workflow commands."""

    name: str
    commands: list[WorkflowCommand]


def discover_workflows() -> list[WorkflowModule]:
    workflows_dir = Path(__file__).parent.parent / "workflows"
    if not workflows_dir.exists():
        return []
    return [
        WorkflowModule(item.name, item)
        for item in workflows_dir.iterdir()
        if item.is_dir() and (item / "__init__.py").exists()
    ]


def discover_workflow_functions(module) -> list[WorkflowCommand]:
    return [
        WorkflowCommand(name, obj)
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and (getattr(obj, "_is_workflow", False) or getattr(obj, "_is_workflow_step", False))
    ]


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def _register_command(app: typer.Typer, func_name: str, func: Callable[..., Any]) -> None:
    """Register a workflow function as a command on a Typer app."""
    sig = inspect.signature(func)
    try:
        type_hints = typing.get_type_hints(func, include_extras=True)
    # lucidlint: ignore swallow an unresolvable annotation is absent; it shows as text
    except (NameError, TypeError):
        # A step's annotation can name a type that only exists at generation
        # time; the CLI then exposes that parameter as plain text.
        type_hints = {}

    sig_params = []
    for param_name, param in sig.parameters.items():
        if param_name in _INTERNAL_PARAMS:
            continue
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
        runner = WorkflowConsole()
        runner.run(_func, workflow_params)

    # Typer reads the signature to build the command's options.
    setattr(  # noqa: B010 — pyrefly has no declared __signature__ attribute on a function
        run_cmd,
        "__signature__",
        inspect.Signature(
            parameters=sig_params,
            return_annotation=inspect.Parameter.empty,
        ),
    )

    app.command(name=_snake_to_kebab(func_name))(run_cmd)


def _discover_class_workflow_methods(module) -> list[WorkflowClass]:
    """Discover exported classes whose public methods are workflow steps."""
    exported = set(getattr(module, "__all__", []))
    groups: list[WorkflowClass] = []
    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj) or name not in exported:
            continue
        commands = [
            WorkflowCommand(mname, mobj)
            for mname, mobj in inspect.getmembers(obj)
            if not mname.startswith("_")
            and (getattr(mobj, "_is_workflow", False) or getattr(mobj, "_is_workflow_step", False))
        ]
        if commands:
            groups.append(WorkflowClass(name, commands))
    return groups


def _build_workflow_sub_app(
    workflow_name: str,
    importer: Callable[[str], Any] = importlib.import_module,
) -> typer.Typer | None:
    """Build a Typer sub-app for a single workflow module.

    Imports the module via ``workflows.{workflow_name}``, discovers
    ``@composite_workflow_step``-decorated module-level functions AND
    classmethods, and registers each.

    Module-level functions become flat subcommand (e.g. ``generate-reviewed-criteria``).
    Class-level methods become a sub-app named for the class's ``__cli_name__``
    when it declares one (e.g. ``workflow create``), otherwise for the class name.

    Returns *None* when the module cannot be imported or contains no
    workflow functions — the caller should skip it silently.
    """
    module_name = f"workflows.{workflow_name}"
    try:
        module = importer(module_name)
    except ImportError:  # lucidlint: ignore swallow an unimportable workflow is skipped; the CLI still lists the rest
        return None

    # Module-level workflow functions (e.g. generate_reviewed_criteria)
    functions = discover_workflow_functions(module)

    # Class-level workflow steps (e.g. Workflow.create)
    class_groups = _discover_class_workflow_methods(module)

    if not functions and not class_groups:
        return None

    sub_app = typer.Typer(help=f"Commands for {workflow_name} workflow")

    # Register module-level functions as flat commands
    for command in functions:
        _register_command(sub_app, command.name, command.func)

    # A module with one workflow class and no module-level functions is the
    # workflow itself: its methods are the commands (e.g. `workflow create`).
    # A class group would only repeat the module name.
    if len(class_groups) == 1 and not functions:
        (group,) = class_groups
        for command in group.commands:
            _register_command(sub_app, command.name, command.func)
        return sub_app

    # Register class-level methods as class sub-apps. The class may declare a
    # user-facing CLI name (e.g. WorkflowSpec -> "workflow") because the spec
    # suffix is an internal term; the command names what it creates.
    for group in class_groups:
        cls = getattr(module, group.name)
        cli_name = getattr(cls, "__cli_name__", group.name)
        class_app = typer.Typer(help=f"Commands for {cli_name}")
        for command in group.commands:
            _register_command(class_app, command.name, command.func)
        sub_app.add_typer(class_app, name=_snake_to_kebab(cli_name))

    return sub_app


def build_cli_app() -> typer.Typer:
    app = typer.Typer(help="Chat Workflow CLI - Generate structured data through LLM conversations")

    for workflow in discover_workflows():
        sub_app = _build_workflow_sub_app(workflow.name)
        if sub_app is None:
            continue
        kebab_name = _snake_to_kebab(workflow.name)
        app.add_typer(sub_app, name=kebab_name)

    return app


app = build_cli_app()

if __name__ == "__main__":
    app()
