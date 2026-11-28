#!/usr/bin/env python3
"""Code generation utilities using libcst.

Provides a ``generate_code()`` function that accepts either:
- A string of Python code (pass-through with formatting)
- Keyword arguments ``class_name`` and ``fields`` to generate a class.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import libcst as cst

__all__ = [
    "FieldSpec",
    "generate_class",
    "generate_code",
    "generate_field",
    "generate_model_validator",
    "generate_workflow_method",
    "import_module",
    "reload_module",
    "verify_code",
]


@dataclass(frozen=True, kw_only=True)
class FieldSpec:
    """One field of a generated class: its name, type and optional description."""

    name: str
    field_type: str
    description: str = ""


def generate_code(
    source: str | None = None,
    *,
    class_name: str | None = None,
    fields: Sequence[FieldSpec] | None = None,
) -> str:
    """Generate Python source code.

    Parameters
    ----------
    source:
        A string of Python code to format / pass through.
    class_name:
        Name of the class to generate (requires ``fields``).
    fields:
        The fields describing the class attributes.

    Returns
    -------
    str
        The generated Python source code.
    """
    if source is not None:
        tree = cst.parse_module(source)
        return tree.code

    if class_name is not None and fields is not None:
        return _generate_model_class(class_name, fields)

    raise TypeError(
        "generate_code() requires either a positional ``source`` argument "
        "or keyword arguments ``class_name`` and ``fields``."
    )


def _generate_model_class(class_name: str, fields: Sequence[FieldSpec]) -> str:
    """Build a class definition using libcst nodes."""
    field_statements: list[cst.BaseSmallStatement] = [
        cst.AnnAssign(
            target=cst.Name(field.name),
            annotation=cst.Annotation(cst.Name(field.field_type)),
        )
        for field in fields
    ]

    # Build the class body as an IndentedBlock containing SimpleStatementLines
    body = cst.IndentedBlock(
        body=[cst.SimpleStatementLine(body=[stmt]) for stmt in field_statements],
    )

    class_def = cst.ClassDef(
        name=cst.Name(class_name),
        body=body,
    )

    module = cst.Module([class_def])
    return module.code


def generate_field(name: str, type_: str, description: str | None = None) -> str:
    """Generate a Pydantic field definition string.

    Parameters
    ----------
    name:
        The field name.
    type_:
        The field type (e.g. ``"str"``, ``"int"``, ``"str | None"``).
    description:
        Optional field description.

    Returns
    -------
    str
        A string like ``name: str = Field(..., description="User name")``.
    """
    if description:
        return f'{name}: {type_} = Field(..., description="{description}")'
    return f"{name}: {type_} = Field(...)"


def generate_workflow_method(class_name: str) -> str:
    """Generate an ``@atomic_workflow_step`` / ``@classmethod`` decorated method.

    Parameters
    ----------
    class_name:
        Name of the class this method belongs to (used in docstring).

    Returns
    -------
    str
        A multi-line string with decorators and method body.
    """
    return (
        "@atomic_workflow_step\n"
        "@classmethod\n"
        "def generate_from_chat(\n"
        "    cls,\n"
        "    context: str,\n"
        "    session = None,\n"
        "):\n"
        f'    """Workflow method for {class_name}."""\n'
        "    ...\n"
    )


def generate_model_validator(rule: str, field_name: str | None = None) -> str:
    """Generate a ``@model_validator`` decorated method.

    Parameters
    ----------
    rule:
        Validation rule string (e.g. ``"name must be non-empty"``). The first
        word is used as the field name unless ``field_name`` is provided.
    field_name:
        Optional explicit field name. If omitted, the first word of ``rule``
        is used.

    Returns
    -------
    str
        A multi-line string with import, decorator, and method body.
    """
    if field_name is None:
        field_name = rule.split()[0]

    return (
        "from chat_workflow import ValidationError\n"
        "\n"
        '@model_validator(mode="after")\n'
        "def validate_business_rules(self):\n"
        f"    if not self.{field_name}:\n"
        f'        raise ValidationError("{rule}")\n'
        "    return self\n"
    )


def _ruff_violations(code: str) -> list[str]:
    """Run ``ruff check`` on a generated module and return its diagnostics."""
    ruff = _find_ruff()
    repo_root = Path.cwd()
    proc = subprocess.run(
        [
            ruff,
            "check",
            "--output-format=concise",
            "--stdin-filename",
            "generated_component.py",
            "-",
        ],
        input=code,
        text=True,
        capture_output=True,
        cwd=repo_root,
        timeout=10,
    )
    if proc.returncode == 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _ruff_format(code: str) -> str:
    """Format a generated module the way the repository formats its own code."""
    ruff = _find_ruff()
    repo_root = Path.cwd()
    proc = subprocess.run(
        [
            ruff,
            "format",
            "--stdin-filename",
            "generated_component.py",
            "-",
        ],
        input=code,
        text=True,
        capture_output=True,
        cwd=repo_root,
        timeout=10,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return code
    return proc.stdout


def verify_code(code: str) -> str:
    """Format generated code, then hold it to ``ruff check``.

    Formatting is deterministic and semantics-preserving. A lint violation is
    raised, never auto-fixed: the caller (or the model that produced the code)
    is the one that has to correct it.
    """
    formatted = _ruff_format(code)
    violations = _ruff_violations(formatted)
    if violations:
        raise RuntimeError(
            "Generated code failed `ruff check`:\n" + "\n".join(violations)
        )
    return formatted


def _find_ruff() -> str:
    """Locate the ``ruff`` executable, searching the venv bin dir first."""
    # Check alongside the current Python interpreter first (venv bin dir)
    venv_bin = os.path.dirname(sys.executable)
    candidate = os.path.join(venv_bin, "ruff")
    if os.path.isfile(candidate):
        return candidate

    # Fall back to PATH lookup
    found = shutil.which("ruff")
    if found is not None:
        return found

    raise RuntimeError("ruff executable not found. Install it with: pip install ruff")


def import_module(path: str):
    """Dynamically import a module by dotted path.

    Parameters
    ----------
    path:
        Dotted module path (e.g. ``"workflows.evaluation_criteria"``).

    Returns
    -------
    module or None
        The imported module, or ``None`` if the path could not be imported.
    """
    try:
        return importlib.import_module(path)
    except ImportError:
        return None


def reload_module(path: str):
    """Reload an already-imported module.

    Parameters
    ----------
    path:
        Dotted module path of an already-imported module.

    Returns
    -------
    module
        The reloaded module.
    """
    return importlib.reload(importlib.import_module(path))


def generate_class(
    name: str,
    fields: Sequence[FieldSpec],
    validation_rules: str | None = None,
) -> str:
    """Generate a complete ``InteractiveEntity`` subclass.

    Parameters
    ----------
    name:
        The class name.
    fields:
        The fields of the generated class.
    validation_rules:
        Optional validation rule string. When provided, a
        ``_validation_rules`` class attribute and a ``@model_validator``
        method are generated.

    Returns
    -------
    str
        Complete Python source code for the class.
    """
    lines: list[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    pydantic_names = ["model_validator"]
    if fields:
        pydantic_names.insert(0, "Field")
    chat_names = ["InteractiveEntity", "atomic_workflow_step"]
    if validation_rules or fields:
        chat_names.insert(1, "ValidationError")
    lines.append(f"from pydantic import {', '.join(pydantic_names)}")
    lines.append("")
    lines.append(f"from chat_workflow import {', '.join(chat_names)}")
    lines.append("")
    lines.append(f"class {name}(InteractiveEntity):")

    if validation_rules:
        lines.append(f'    _validation_rules: str = "{validation_rules}"')
        lines.append("")

    lines.extend(_field_lines(fields))

    _method_header(
        lines,
        ['    @model_validator(mode="after")'],
        "    def validate_business_rules(self):",
    )

    if validation_rules:
        lines.extend(_empty_guard(validation_rules.split()[0], validation_rules))
    elif fields:
        first_field = fields[0].name
        lines.extend(_empty_guard(first_field, f"{first_field} must be non-empty"))
    else:
        lines.append("        pass")

    lines.append("        return self")
    _method_header(
        lines,
        ["    @atomic_workflow_step", "    @classmethod"],
        "    def generate_from_chat(",
    )
    lines.append("        cls,")
    lines.append("        context: str,")
    lines.append("        session = None,")
    lines.append("    ):")
    lines.append(f'        """{name} workflow."""')
    lines.append("        ...")
    lines.append("")

    return "\n".join(lines)


def _field_lines(fields: Sequence[FieldSpec]) -> list[str]:
    return [
        f'    {field.name}: {field.field_type} = Field(..., description="{field.description}")'
        if field.description
        else f"    {field.name}: {field.field_type} = Field(...)"
        for field in fields
    ]


def _empty_guard(field: str, message: str) -> list[str]:
    return [
        f"        if not self.{field}:",
        f'            raise ValidationError("{message}")',
    ]


def _method_header(lines: list[str], decorators: list[str], signature: str) -> None:
    lines.append("")
    lines.extend(decorators)
    lines.append(signature)
