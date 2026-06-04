#!/usr/bin/env python3
"""Code generation utilities using libcst.

Provides a ``generate_code()`` function that accepts either:
- A string of Python code (pass-through with formatting)
- Keyword arguments ``class_name`` and ``fields`` to generate a class.
"""

from __future__ import annotations

import os

import libcst as cst


def generate_code(
    source: str | None = None,
    *,
    class_name: str | None = None,
    fields: list[tuple[str, str]] | None = None,
) -> str:
    """Generate Python source code.

    Parameters
    ----------
    source:
        A string of Python code to format / pass through.
    class_name:
        Name of the class to generate (requires ``fields``).
    fields:
        List of ``(field_name, field_type)`` tuples describing the class
        attributes.

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


def _generate_model_class(class_name: str, fields: list[tuple[str, str]]) -> str:
    """Build a class definition using libcst nodes."""
    field_statements: list[cst.BaseSmallStatement] = []

    for field_name, field_type in fields:
        ann = cst.AnnAssign(
            target=cst.Name(field_name),
            annotation=cst.Annotation(cst.Name(field_type)),
        )
        field_statements.append(ann)

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
    """Generate an ``@atomic_workflow`` / ``@classmethod`` decorated method.

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
        "@atomic_workflow\n"
        "@classmethod\n"
        "def generate_from_chat(\n"
        "    cls,\n"
        "    context: str,\n"
        "    max_turns: int = 10,\n"
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


def verify_code(code: str, max_attempts: int = 3) -> str:
    """Verify generated code passes ruff linting and formatting checks.

    Runs ``ruff format --check`` and ``ruff check`` on the code. If either
    fails, attempts to auto-fix using ``ruff format`` and ``ruff check --fix``.
    Loops until the code passes or ``max_attempts`` is reached.

    Parameters
    ----------
    code:
        The Python source code to verify.
    max_attempts:
        Maximum number of fix attempts (default: 3).

    Returns
    -------
    str
        The cleaned, lint-free Python source code.

    Raises
    ------
    RuntimeError
        If the code cannot be fixed after ``max_attempts``.
    """
    import os
    import subprocess
    import tempfile

    ruff_path = _find_ruff()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    fmt_result = None
    lint_result = None

    try:
        for _ in range(max_attempts):
            # Check formatting
            fmt_result = subprocess.run(
                [ruff_path, "format", "--check", tmp_path],
                capture_output=True,
                text=True,
            )

            if fmt_result.returncode != 0:
                subprocess.run(
                    [ruff_path, "format", tmp_path],
                    capture_output=True,
                    text=True,
                )
                continue

            # Check linting — no rules ignored. If the generated code violates
            # ruff rules, the prompt must be improved to generate cleaner code.
            lint_result = subprocess.run(
                [ruff_path, "check", tmp_path],
                capture_output=True,
                text=True,
            )

            if lint_result.returncode != 0:
                subprocess.run(
                    [ruff_path, "check", "--fix", tmp_path],
                    capture_output=True,
                    text=True,
                )
                continue

            # Both passed — read and return cleaned code
            with open(tmp_path) as f:
                return f.read()

        # All attempts exhausted
        parts = [f"Code failed ruff checks after {max_attempts} attempts."]
        if fmt_result is not None:
            parts.append("--- format --check ---")
            parts.append(fmt_result.stdout)
            parts.append(fmt_result.stderr)
        if lint_result is not None:
            parts.append("--- check ---")
            parts.append(lint_result.stdout)
            parts.append(lint_result.stderr)
        raise RuntimeError("\n".join(parts))

    finally:
        os.unlink(tmp_path)


def _find_ruff() -> str:
    """Locate the ``ruff`` executable, searching the venv bin dir first."""
    import shutil
    import sys

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
    import importlib

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
    import importlib

    return importlib.reload(importlib.import_module(path))


def generate_class(
    name: str,
    fields: list[dict[str, str]],
    validation_rules: str | None = None,
) -> str:
    """Generate a complete ``InteractiveEntity`` subclass.

    Parameters
    ----------
    name:
        The class name.
    fields:
        List of dicts with keys ``"name"``, ``"type"``, and optionally
        ``"desc"``.
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
    lines.append("from pydantic import BaseModel, Field, model_validator")
    lines.append("from chat_workflow import InteractiveEntity, ValidationError, atomic_workflow")
    lines.append("")
    lines.append(f"class {name}(InteractiveEntity):")

    if validation_rules:
        lines.append(f'    _validation_rules: str = "{validation_rules}"')
        lines.append("")

    for field in fields:
        field_desc = field.get("desc", "")
        if field_desc:
            lines.append(f'    {field["name"]}: {field["type"]} = Field(..., description="{field_desc}")')
        else:
            lines.append(f"    {field['name']}: {field['type']} = Field(...)")

    lines.append("")
    lines.append('    @model_validator(mode="after")')
    lines.append("    def validate_business_rules(self):")

    if validation_rules:
        v_field = validation_rules.split()[0]
        lines.append(f"        if not self.{v_field}:")
        lines.append(f'            raise ValidationError("{validation_rules}")')
    elif fields:
        first_field = fields[0]["name"]
        lines.append(f"        if not self.{first_field}:")
        lines.append(f'            raise ValidationError("{first_field} must be non-empty")')
    else:
        lines.append("        pass")

    lines.append("        return self")
    lines.append("")
    lines.append("    @atomic_workflow")
    lines.append("    @classmethod")
    lines.append("    def generate_from_chat(")
    lines.append("        cls,")
    lines.append("        context: str,")
    lines.append("        max_turns: int = 10,")
    lines.append("        session = None,")
    lines.append("    ):")
    lines.append(f'        """{name} workflow."""')
    lines.append("        ...")
    lines.append("")

    return "\n".join(lines)
