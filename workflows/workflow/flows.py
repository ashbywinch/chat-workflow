"""Workflow management commands.

Usage: chat-workflow workflow create
"""

from __future__ import annotations

from pathlib import Path

from chat_workflow import Session, composite_workflow
from chat_workflow.code_generator import (
    generate_class,
    import_module,
    reload_module,
    verify_code,
)


def _sanitize_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    if name and not name[0].isalpha():
        name = "workflow_" + name
    return name.lower()


def _to_class_name(name: str) -> str:
    parts = name.split("_")
    return "".join(part.capitalize() for part in parts)


@composite_workflow
def create(*, session: Session) -> None:
    io = session.io
    io.echo("")
    io.echo("=== Create Workflow Wizard ===")
    io.echo("I'll help you create a new workflow step by step.")
    io.echo("")

    while True:
        name_raw = io.prompt("What would you like to name your workflow?")
        if not name_raw.strip():
            io.echo("Name cannot be empty. Please try again.")
            continue

        name = _sanitize_name(name_raw)
        class_name = _to_class_name(name)
        io.echo(f"Creating workflow '{class_name}' in workflows/{name}/")

        fields: list[dict[str, str]] = []
        io.echo("")
        io.echo("Now let's define the fields for your workflow.")
        io.echo("Enter one field at a time. Leave the name empty to finish.")

        while True:
            field_name = io.prompt("  Field name (or empty to finish)", default="")
            if not field_name.strip():
                if not fields:
                    io.echo("  You need at least one field. Please add a field.")
                    continue
                break

            field_name = _sanitize_name(field_name)
            field_type = io.prompt(f"  Type for '{field_name}' (str, int, float, list, etc.)")
            if not field_type.strip():
                field_type = "str"
            field_desc = io.prompt(f"  Description for '{field_name}'", default="")

            fields.append(
                {
                    "name": field_name,
                    "type": field_type.strip(),
                    "desc": field_desc.strip(),
                }
            )
            io.echo(f"  Added: {field_name}: {field_type.strip()}")

        io.echo("")
        io.echo("Validation rules help ensure data quality.")
        io.echo("Example: 'name must be non-empty' or 'count must be between 1 and 100'")
        validation_rules = io.prompt("  Validation rules (empty for none)", default="")

        io.echo("")
        io.echo("Generating code...")
        code = generate_class(
            name=class_name,
            fields=fields,
            validation_rules=validation_rules.strip() or None,
        )

        io.echo("Verifying code quality...")
        try:
            clean_code = verify_code(code)
        except RuntimeError as e:
            io.echo(f"Code verification failed: {e}")
            retry = io.prompt("Try creating a different workflow? (y/n)")
            if retry.strip().lower().startswith("y"):
                continue
            io.echo("Exiting. The generated code had quality issues.")
            return

        io.echo("Saving workflow files...")
        workflow_dir = Path(__file__).resolve().parent.parent / name
        workflow_dir.mkdir(exist_ok=True)

        (workflow_dir / "models.py").write_text(clean_code)
        (workflow_dir / "__init__.py").write_text(
            f"from .models import {class_name}\n\n__all__ = ['{class_name}']\n"
        )

        io.echo("Loading new workflow...")
        mod = import_module(f"workflows.{name}")
        if mod:
            reload_module(f"workflows.{name}")

        io.echo("")
        io.echo(f"Workflow '{class_name}' created successfully!")
        io.echo(f"Location: workflows/{name}/")
        io.echo(f"To run: chat-workflow {name.replace('_', '-')} --context 'your context'")

        io.echo("")
        action = io.prompt(
            "What would you like to do? (create another / finish)", default="finish"
        )
        action = action.strip().lower()

        if action.startswith("c"):
            io.echo("")
            io.echo("Let's create another workflow!")
            continue
        else:
            io.echo("")
            io.echo("Goodbye! Happy workflow building!")
            return
