from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, ClassVar, override

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import LLMValidated

from .design_spec import ComponentDesignSpec

_BASE_MODEL_SOURCES = frozenset({"pydantic", "chat_workflow.mixins"})

# Phrases that restate the response protocol. The envelope and Instructor own
# these; a generated step docstring must describe the conversation instead.
_PROTOCOL_PHRASES = (
    "return json",
    "return only a json",
    "put the user-facing text in",
    "put the text in `message`",
    "never reply with plain text",
    "set `intent`",
    "set intent to",
)


def _basemodel_class_names(tree: ast.AST) -> set[str]:
    """Names bound to BaseModel (or a mixin that subclasses it)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _BASE_MODEL_SOURCES:
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BASE_MODEL_SOURCES:
                    names.add(alias.asname or alias.name.rsplit(".", 1)[-1])
    return names


def _join_source_lines(value: Any) -> Any:
    """Accept the module as lines or as one string.

    A model occasionally puts several lines into one array element as
    newline escapes. The contract is one string per line, so split them
    mechanically instead of spending a retry on the escape.
    """
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            lines.extend(str(item).split("\n"))
        return "\n".join(lines)
    return value


class SourceLines(str):
    """A module stored as text but exchanged with the model as lines.

    A long module full of escaped newlines is where the provider's escape
    corruption bites: a dropped character after a backslash silently merges
    two lines. Asking for one string per line removes the escapes entirely,
    while every consumer still sees ordinary source text.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            _join_source_lines, core_schema.str_schema()
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
        return {"type": "array", "items": {"type": "string"}}


def _column_to_index(line: str, column: int) -> int:
    """Convert an AST column offset into a character index in ``line``.

    ``ast`` reports column offsets in UTF-8 bytes. Slicing a ``str`` with them
    lands in the wrong place as soon as the line holds a non-ASCII character —
    an em dash in a description is enough to splice code into the next line.
    """
    if column <= 0:
        return 0
    return len(line.encode("utf-8")[:column].decode("utf-8", errors="ignore"))


def _absolute_index(code: str, lineno: int, column: int) -> int:
    """The character index of an AST position in the whole module."""
    lines = code.splitlines(keepends=True)
    return sum(len(line) for line in lines[: lineno - 1]) + _column_to_index(
        lines[lineno - 1], column
    )


_SPLIT_DESCRIPTION = re.compile(
    r'(description\s*=\s*"[^"]*")\s*,\s*\n(\s*)(")'
)


_TRAILING_GARBAGE = frozenset({"}", ")", "]", "to", "the", "and"})


def _trim_trailing_garbage(code: str) -> str:
    """Drop stray lines a model leaves after a complete module."""
    lines = code.splitlines()
    changed = False
    while lines and lines[-1].strip() in _TRAILING_GARBAGE:
        lines.pop()
        changed = True
    if not changed:
        return code
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def _join_split_descriptions(code: str) -> str:
    """Join a description string the model split into two positional strings.

    ``description="one part",`` followed by ``" another part"`` is a syntax
    error: the continuation is parsed as a positional argument after a keyword.
    Adjacent string literals are valid, so removing the comma is enough.
    """
    return _SPLIT_DESCRIPTION.sub(r"\1\n\2\3", code)


def _ensure_min_length(code: str, tree: ast.AST) -> str:
    """Add ``min_length=1`` to required ``str`` fields that lack it.

    A required string with no length constraint accepts the empty string, which
    makes the artifact categorically invalid. Pydantic supports this constraint
    directly, so it is filled in mechanically instead of relying on the model
    to restate it. Fields with an existing ``min_length`` (including
    ``min_length=0``) and optional fields are left alone.
    """
    base_names = _basemodel_class_names(tree)
    if not base_names:
        return code

    insertions: list[int] = []
    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue
        if not any(isinstance(base, ast.Name) and base.id in base_names for base in cls_node.bases):
            continue
        for node in cls_node.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            annotation = node.annotation
            annotation_name = (
                annotation.id
                if isinstance(annotation, ast.Name)
                else annotation.value
                if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
                else None
            )
            if annotation_name != "str":
                continue
            call = node.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "Field"
            ):
                continue
            # Required either by an explicit ``...`` first argument or by
            # having no default at all (``Field(description=...)``).
            has_ellipsis = bool(
                call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value is Ellipsis
            )
            has_default = any(
                kw.arg in ("default", "default_factory") for kw in call.keywords
            )
            if not has_ellipsis and (call.args or has_default):
                continue
            if any(kw.arg == "min_length" for kw in call.keywords):
                continue
            if call.keywords:
                last_arg: ast.expr = call.keywords[-1].value
            elif call.args:
                last_arg = call.args[-1]
            else:
                continue
            if last_arg.end_lineno is None or last_arg.end_col_offset is None:
                continue
            insertions.append(
                _absolute_index(code, last_arg.end_lineno, last_arg.end_col_offset)
            )

    for pos in sorted(insertions, reverse=True):
        code = code[:pos] + ", min_length=1" + code[pos:]
    return code


def _escaped_length(text: str) -> int:
    """The width a string occupies once ``json.dumps`` has escaped it.

    A non-ASCII character costs one character in the source but six in the
    rendered literal (``\\u2014``), so measuring the raw text overruns the
    line limit.
    """
    return len(json.dumps(text))


def _split_description(
    text: str, width: int, measure: Callable[[str], int] = len
) -> list[str]:
    """Split a description into lines that fit, preserving spaces at joins."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if measure(candidate) > width and current:
            lines.append(current + " ")
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _wrap_long_field_calls(code: str, tree: ast.AST) -> str:
    """Split single-line ``Field(...)`` calls that exceed the line limit.

    Models reliably keep long descriptions on one line even after repeated
    E501 feedback. Wrapping at the argument boundaries is mechanical, so it is
    done here instead of spending retries on formatting.
    """
    lines = code.splitlines(keepends=True)
    replacements: list[tuple[int, int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Field"
        ):
            continue
        if node.lineno != node.end_lineno:
            continue
        if node.end_col_offset is None:
            continue
        line = lines[node.lineno - 1].rstrip("\n")
        if len(line) <= 120:
            continue
        arguments: list[str] = []
        indent = " " * node.col_offset
        argument_indent = " " * (node.col_offset + 4)
        for argument in [*node.args, *node.keywords]:
            if isinstance(argument, ast.keyword):
                if argument.arg is None:
                    arguments = []
                    break
                segment = ast.get_source_segment(code, argument.value)
                if segment is None:
                    arguments = []
                    break
                single = f"{argument.arg}={segment}"
                if len(argument_indent + single) <= 120 or not (
                    isinstance(argument.value, ast.Constant)
                    and isinstance(argument.value.value, str)
                ):
                    arguments.append(single)
                    continue
                width = max(20, 120 - len(argument_indent) - 4 - 3)
                chunks = _split_description(
                    argument.value.value, width, _escaped_length
                )
                inner = "\n".join(f"    {json.dumps(chunk)}" for chunk in chunks)
                arguments.append(f"{argument.arg}=(\n{inner}\n)")
            else:
                segment = ast.get_source_segment(code, argument)
                if segment is None:
                    arguments = []
                    break
                arguments.append(segment)
        if not arguments:
            continue
        rendered = [
            f"{argument_indent}{argument.replace(chr(10), chr(10) + argument_indent)}"
            for argument in arguments
        ]
        wrapped = "Field(\n" + ",\n".join(rendered) + ",\n" + indent + ")"
        replacements.append((node.lineno - 1, node.col_offset, node.end_col_offset, wrapped))
    for line_number, start, end, wrapped in sorted(replacements, reverse=True):
        line = lines[line_number]
        newline = "\n" if line.endswith("\n") else ""
        content = line.rstrip("\n")
        start_index = _column_to_index(content, start)
        end_index = _column_to_index(content, end)
        lines[line_number] = (
            content[:start_index] + wrapped + content[end_index:] + newline
        )
    return "".join(lines)


def _wrap_long_field_strings(code: str, tree: ast.AST) -> str:
    """Wrap long string constants inside already multi-line ``Field(...)`` calls.

    Models often wrap the call but leave one implicit-concatenation segment
    longer than the line limit. The AST value is the concatenated string, so
    rebuilding it at word boundaries is safe and preserves the text.
    """
    lines = code.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Field"
        ):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if not (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.end_lineno is not None
                and value.end_col_offset is not None
            ):
                continue
            start_line = value.lineno - 1
            end_line = value.end_lineno - 1
            if not any(
                len(lines[index].rstrip("\n")) > 120
                for index in range(start_line, end_line + 1)
            ):
                continue
            # `_split_description` keeps a trailing space on every line but the
            # last so concatenation preserves the text; leave room for it.
            width = max(20, 120 - value.col_offset - 3)
            chunks = _split_description(value.value, width, _escaped_length)
            indent = " " * value.col_offset
            rendered = json.dumps(chunks[0])
            for chunk in chunks[1:]:
                rendered += "\n" + indent + json.dumps(chunk)
            start = _absolute_index(code, start_line + 1, value.col_offset)
            end = _absolute_index(code, end_line + 1, value.end_col_offset)
            replacements.append((start, end, rendered))
    for start, end, rendered in sorted(replacements, reverse=True):
        code = code[:start] + rendered + code[end:]
    return code


def _wrap_long_list_strings(code: str, tree: ast.AST) -> str:
    """Wrap long string elements of lists and tuples at word boundaries.

    ``_validation_rules`` entries are list elements, not ``Field`` keyword
    arguments, so the field wrapper does not reach them. Models keep writing
    one long rule line; splitting it mechanically avoids a retry.
    """
    lines = code.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        for element in node.elts:
            if not (
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                and element.end_lineno is not None
                and element.end_col_offset is not None
            ):
                continue
            start_line = element.lineno - 1
            end_line = element.end_lineno - 1
            if not any(
                len(lines[index].rstrip("\n")) > 120
                for index in range(start_line, end_line + 1)
            ):
                continue
            width = max(20, 120 - element.col_offset - 3)
            chunks = _split_description(element.value, width, _escaped_length)
            indent = " " * element.col_offset
            rendered = json.dumps(chunks[0])
            for chunk in chunks[1:]:
                rendered += "\n" + indent + json.dumps(chunk)
            start = _absolute_index(code, start_line + 1, element.col_offset)
            end = _absolute_index(code, end_line + 1, element.end_col_offset)
            replacements.append((start, end, rendered))
    for start, end, rendered in sorted(replacements, reverse=True):
        code = code[:start] + rendered + code[end:]
    return code


def _fix_field_validator_self(code: str, tree: ast.AST) -> str:
    """Rename a field validator's first argument from ``self`` to ``cls``.

    Pydantic's ``@field_validator`` is a classmethod: it receives the class,
    and a validator that takes ``self`` raises at import time. The rename is
    mechanical and safe.
    """
    replacements: list[tuple[int, int, int]] = []
    insertions: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_field_validator = any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "field_validator"
            for decorator in node.decorator_list
        )
        if not is_field_validator:
            continue
        # Ruff only knows why the first argument is not `self` when the
        # classmethod decorator is visible, so make it explicit.
        has_classmethod = any(
            isinstance(decorator, ast.Name) and decorator.id == "classmethod"
            for decorator in node.decorator_list
        )
        if not has_classmethod and node.decorator_list:
            last = max(
                decorator.end_lineno or decorator.lineno
                for decorator in node.decorator_list
            )
            insertions.append((last, " " * node.col_offset + "@classmethod\n"))
        for argument in [*node.args.args, *node.args.kwonlyargs]:
            if argument.arg == "self" and argument.end_col_offset is not None:
                replacements.append(
                    (argument.lineno, argument.col_offset, argument.end_col_offset)
                )
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Name)
                and child.id == "self"
                and child.end_col_offset is not None
            ):
                replacements.append(
                    (child.lineno, child.col_offset, child.end_col_offset)
                )
    lines = code.splitlines(keepends=True)
    for lineno, start, end in sorted(replacements, reverse=True):
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:start] + "cls" + line[end:]
    for after_line, text in sorted(insertions, reverse=True):
        lines.insert(after_line, text)
    return "".join(lines)


def _add_raise_from_none(code: str, tree: ast.AST) -> str:
    """Make a bare ``raise`` inside an ``except`` clause explicit.

    Ruff's B904 requires the cause to be stated. A validator that turns one
    error into another almost always means ``from None``.
    """
    insertions: list[int] = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        for node in ast.walk(handler):
            if not isinstance(node, ast.Raise):
                continue
            if node.exc is None or node.cause is not None:
                continue
            if node.end_lineno is None or node.end_col_offset is None:
                continue
            insertions.append(
                _absolute_index(code, node.end_lineno, node.end_col_offset)
            )
    for position in sorted(insertions, reverse=True):
        code = code[:position] + " from None" + code[position:]
    return code


def _ruff_executable() -> str | None:
    """Return the ruff binary for this environment, if present."""
    venv_ruff = Path(sys.executable).with_name("ruff")
    if venv_ruff.exists():
        return str(venv_ruff)
    return shutil.which("ruff")


def _ruff_violations(code: str) -> list[str]:
    """Run ``ruff check`` on the generated module and return its diagnostics.

    Uses the repository's ``pyproject.toml`` so generated code is held to the
    same standard as the codebase. When ruff is unavailable, falls back to the
    line-length rule only; syntax is checked separately by the caller.
    """
    ruff = _ruff_executable()
    if ruff is None:
        return [
            f"generated_component.py:{number}: E501 Line too long "
            f"({len(line)} > 120)"
            for number, line in enumerate(code.splitlines(), 1)
            if len(line) > 120
        ]

    repo_root = Path(__file__).resolve().parents[2]
    try:
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
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"ruff could not run: {exc}"]

    if proc.returncode == 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _ruff_format(code: str) -> str:
    """Format the module the way the repository formats its own code.

    ``ruff format`` wraps long expressions at the line limit, which is what a
    model gets wrong most often when it joins a condition onto one line. It is
    deterministic, so running it is cheaper than a retry. Returns the original
    code when ruff is unavailable or the module does not parse.
    """
    ruff = _ruff_executable()
    if ruff is None:
        return code

    repo_root = Path(__file__).resolve().parents[2]
    try:
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
    except (OSError, subprocess.TimeoutExpired):
        return code
    if proc.returncode != 0 or not proc.stdout.strip():
        return code
    return proc.stdout


def _ruff_fix(code: str) -> str:
    """Apply ruff's safe automatic fixes and return the rewritten module.

    Import order, unused imports and modern type syntax are mechanical; fixing
    them here keeps model retries for judgment calls such as wrapping long
    lines. Returns the original code when ruff is unavailable.
    """
    ruff = _ruff_executable()
    if ruff is None:
        return code

    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            [
                ruff,
                "check",
                "--fix",
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
    except (OSError, subprocess.TimeoutExpired):
        return code

    # ruff writes the rewritten module to stdout; diagnostics go to stderr.
    return proc.stdout if proc.stdout.strip() else code


def _annotation_names(node: ast.AST | None) -> set[str]:
    """Return every class name mentioned in an annotation.

    Handles ``Model``, ``list[Model]``, ``Model | None`` and string forward
    references (including ``list["Model"]``).
    """
    if node is None:
        return set()

    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                inner = ast.parse(sub.value, mode="eval").body
            except SyntaxError:
                continue
            names |= _annotation_names(inner)
    return names


def _class_name(cls: object) -> str:
    """Return the name of a runtime class or an AST class node."""
    if isinstance(cls, type):
        return cls.__name__
    return getattr(cls, "name", "")


def _uses_llm_validated(main_cls: object, tree: ast.AST) -> bool:
    """True when the primary model inherits ``LLMValidated``."""
    if isinstance(main_cls, type) and issubclass(main_cls, LLMValidated):
        return True
    name = _class_name(main_cls)
    node = next(
        (
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, ast.ClassDef) and candidate.name == name
        ),
        None,
    )
    if node is None:
        return False
    return any(
        isinstance(base, ast.Name) and base.id == "LLMValidated"
        for base in node.bases
    )


def _returned_model_names(
    workflow_methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
    base_names: set[str],
) -> set[str]:
    """Return every BaseModel the workflow steps declare as a return type."""
    returned: set[str] = set()
    for node in workflow_methods:
        returned |= _annotation_names(node.returns) & base_names
    return returned


def _resolve_primary_model(
    workflow_methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
    returned: set[str],
    class_nodes: dict[str, ast.ClassDef],
    base_names: set[str],
) -> str | None:
    """Return the artifact the component owns, if its methods identify one.

    A component may have several workflow steps — create, update, and so on.
    They normally return the same model, which is the artifact. A method may
    also return an auxiliary type the artifact uses. Two models that are
    neither the same nor reachable from one another belong to different
    components.
    """
    if not returned:
        return None
    if len(returned) == 1:
        return next(iter(returned))

    reachable = {
        name: _reachable_model_names(name, class_nodes, base_names)
        for name in returned
    }
    roots = [
        name
        for name in sorted(returned)
        if all(other in reachable[name] for other in returned)
    ]
    if roots:
        return roots[0]

    # No model contains the others; the one most methods return is the
    # artifact, as in a create/update pair that also returns a helper type.
    counts = {
        name: sum(
            1 for node in workflow_methods if name in _annotation_names(node.returns)
        )
        for name in returned
    }
    best = max(counts.values())
    leaders = [name for name, count in counts.items() if count == best]
    if len(leaders) == 1:
        return leaders[0]
    raise ValidationError(
        "Workflow steps return independent BaseModel types "
        f"({', '.join(sorted(returned))}). A component owns one artifact; "
        "its methods may return that artifact or a type it uses, but not a "
        "second independent artifact."
    )


def _reachable_model_names(
    primary: str,
    class_nodes: dict[str, ast.ClassDef],
    base_names: set[str],
) -> set[str]:
    """Return the primary model plus every BaseModel reachable through fields."""
    reachable: set[str] = {primary}
    queue = [primary]
    while queue:
        name = queue.pop()
        node = class_nodes.get(name)
        if node is None:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            for reference in _annotation_names(statement.annotation) & base_names:
                if reference not in reachable:
                    reachable.add(reference)
                    queue.append(reference)
    return reachable


def _workflow_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool]:
    """Return ``(is_atomic, is_chat_off)`` for a workflow step."""
    is_atomic = False
    is_chat_off = False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        else:
            name = None
        if name == "atomic_workflow_step":
            is_atomic = True
        if isinstance(decorator, ast.Call):
            for kw in decorator.keywords:
                if (
                    kw.arg == "chat"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    is_chat_off = True
    return is_atomic, is_chat_off


def _is_model_validator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the method is decorated with ``@model_validator``."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "model_validator":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "model_validator":
            return True
    return False


def _model_validator_field_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    """Return the field name passed to ``@model_validator('field')``, if any.

    ``model_validator`` does not take a field name. Models reach for it when
    they mean ``field_validator``; the resulting code parses but raises at
    import time, so it must be rejected deterministically.
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        if not (isinstance(target, ast.Name) and target.id == "model_validator"):
            continue
        for argument in decorator.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                return argument.value
    return None


def _sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, keeping abbreviations intact enough.

    Punctuation is the boundary that matters here: a run that only exists
    because two sentences were concatenated is not a repeated phrase.
    """
    return [part for part in re.split(r"(?<=[.!?;:])\s+|\n+", text) if part.strip()]


def _shared_phrase(first: str, second: str, size: int = 6) -> str | None:
    """Return a shared run of *size* consecutive words, or ``None``.

    The run must sit inside a single sentence of each text. Comparing whole
    texts lets punctuation-stripped fragments such as "the user. 'Owner'"
    masquerade as a repeated phrase, which made the check fire on hand-written
    documentation that merely shared a sentence opening.
    """
    second_sentences = [_words(sentence) for sentence in _sentences(second)]
    grams: set[str] = set()
    for sentence_words in second_sentences:
        for i in range(len(sentence_words) - size + 1):
            grams.add(" ".join(sentence_words[i : i + size]))
    if not grams:
        return None
    for sentence in _sentences(first):
        first_words = _words(sentence)
        if len(first_words) < size:
            continue
        for i in range(len(first_words) - size + 1):
            phrase = " ".join(first_words[i : i + size])
            if phrase in grams:
                return phrase
    return None


def _words(text: str) -> list[str]:
    """Normalise a sentence to lowercase alphanumeric words."""
    return re.findall(r"[a-z0-9]+", text.lower())


_FRAMEWORK_IMPORT_ALIASES: dict[str, str] = {
    # The package was reorganised; the old paths still appear in the wild.
    "chat_workflow.workflow": "chat_workflow",
    "chat_workflow.workflow.decorators": "chat_workflow",
}


def _canonical_framework_imports(code: str) -> str:
    """Rewrite a framework import path that has an exact replacement."""
    for legacy, canonical in _FRAMEWORK_IMPORT_ALIASES.items():
        if legacy not in code:
            continue
        code = re.sub(
            rf"(from\s+){re.escape(legacy)}\b",
            lambda match, canonical=canonical: match.group(1) + canonical,
            code,
        )
    return code


def _join_tuple_descriptions(code: str, tree: ast.AST) -> str:
    """Collapse a tuple of strings in ``description=`` into one string.

    A model that wraps a description across lines sometimes writes
    ``description=("first part", "second part")``. That is a tuple, not
    implicit concatenation: pydantic accepts it silently and every reader of
    the field gets a tuple where a sentence was meant.
    """
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Field"
        ):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg != "description" or not isinstance(value, ast.Tuple):
                continue
            parts: list[str] = []
            for element in value.elts:
                if not (
                    isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                ):
                    parts = []
                    break
                parts.append(element.value)
            if not parts:
                continue
            if value.end_lineno is None or value.end_col_offset is None:
                continue
            start = _absolute_index(code, value.lineno, value.col_offset)
            end = _absolute_index(code, value.end_lineno, value.end_col_offset)
            joined = " ".join(parts)
            replacements.append((start, end, json.dumps(joined)))
    for start, end, rendered in sorted(replacements, reverse=True):
        code = code[:start] + rendered + code[end:]
    return code


def _unresolvable_imports(tree: ast.AST) -> list[str]:
    """Return module paths that cannot be resolved in this environment.

    Generated component code is a single module. A model that invents a
    dependency (for example ``pydantic_ai``) or a framework submodule that does
    not exist (``chat_workflow.workflow``) produces code that cannot be
    imported. This is a resolution check, not an allowlist: any package that
    is actually installed is accepted. Relative imports are allowed for
    framework source files.
    """
    module_paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_paths.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            module_paths.add(node.module)

    unresolved: list[str] = []
    for path in sorted(module_paths):
        try:
            if importlib.util.find_spec(path) is None:
                unresolved.append(path)
        except (ImportError, ModuleNotFoundError, ValueError):
            unresolved.append(path)
    return unresolved


class ComponentSourceCode(LLMValidated):
    """Python source code for one chat-workflow business component."""

    _validation_rules: ClassVar[list[str]] = [
        "The primary model inherits LLMValidated and declares at least one "
        "_validation_rules entry for a non-deterministic domain business rule "
        "a domain expert would care about. Mechanical field constraints and "
        "@model_validator methods alone are not enough.",
        "For a chat-on atomic workflow step (an @atomic_workflow_step without "
        "chat=False), the workflow docstring describes the conversation — "
        "what to ask, in what order, what to watch for, and when to stop — "
        "not the artifact or its fields. A chat-off atomic workflow step or a "
        "composite orchestrator has no user conversation; its docstring may "
        "be empty or documentation.",
        "REJECT if the workflow docstring restates the class docstring or "
        "enumerates the model's fields. A conversation-only docstring may "
        "name the artifact once.",
        "In the generated model, mechanical constraints (non-empty, numeric "
        "range, pattern) use Pydantic field constraints or @model_validator. "
        "Reserve the generated model's _validation_rules for domain-quality "
        "criteria that cannot be expressed structurally — specificity, domain "
        "fit, and excellence by the user's standards.",
        "REJECT a _validation_rules entry that repeats a field constraint or "
        "a @model_validator; for example, do not list 'blockers must have an "
        "unblocker' when the validator already checks it.",
        "REJECT if the model mixes the artifact with cross-cutting concerns "
        "that are not part of it — for example notification, scheduling, "
        "storage, tenancy, audit, or delivery state — or with an unrelated "
        "sub-artifact or utility. The examples are illustrative, not "
        "exhaustive. All fields and helpers must serve the one artifact.",
        "REJECT if the class docstring describes the model's fields, "
        "structure, or validation rules.",
        "A generated step's docstring describes that step's conversation "
        "procedure only. It does not define a greeting or a closing, and the "
        "generated code does not define a SessionComponent or wrap a step in "
        "one: the framework's session wrapper opens and closes the "
        "conversation.",
        "REJECT any generated docstring or prompt that states the response "
        "format — JSON, intent, message, result, or 'never reply with plain "
        "text'. The response schema owns the protocol.",
        "If a generated step declares a conversation checklist, it complements "
        "the step's docstring rather than paraphrasing it, and each rule in the "
        "declared checklist names a behaviour the judge can falsify. A step "
        "with no declared checklist is not rejected for that absence.",
    ]
    _validation_rules_heading: ClassVar[str] = "Generated code validation rules"

    @override
    def validate_deterministic_rules(self) -> None:
        """Programmatic validation before LLM-judged rules.

        Syntax check, runtime BaseModel subclass check (resolves indirect
        inheritance like LLMValidated → BaseModel), public workflow step,
        return type check, and business validation presence check.
        """
        code = self.code
        repaired = _trim_trailing_garbage(_join_split_descriptions(code))
        if repaired != code:
            self.code = SourceLines(repaired)
            code = repaired

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            lines = code.splitlines()
            offending = ""
            if e.lineno and 0 < e.lineno <= len(lines):
                offending = lines[e.lineno - 1]
            raise ValidationError(
                f"Generated code is not valid Python: {e}. "
                f"Offending line {e.lineno}: {offending!r}. "
                "Write plain ASCII Python: no Markdown fences, full-width "
                "characters, or non-ASCII punctuation."
            ) from e

        # Required string fields must carry a length constraint. The schema can
        # express this, so fill it in rather than spending judge retries on it.
        normalised = _ensure_min_length(code, tree)
        if normalised != code:
            self.code = SourceLines(normalised)
            code = normalised
            tree = ast.parse(code)

        wrapped = _wrap_long_field_calls(code, tree)
        if wrapped != code:
            self.code = SourceLines(wrapped)
            code = wrapped
            tree = ast.parse(code)

        wrapped = _wrap_long_field_strings(code, tree)
        if wrapped != code:
            self.code = SourceLines(wrapped)
            code = wrapped
            tree = ast.parse(code)

        wrapped = _wrap_long_list_strings(code, tree)
        if wrapped != code:
            self.code = SourceLines(wrapped)
            code = wrapped
            tree = ast.parse(code)

        fixed_self = _fix_field_validator_self(code, tree)
        if fixed_self != code:
            self.code = SourceLines(fixed_self)
            code = fixed_self
            tree = ast.parse(code)

        explicit = _add_raise_from_none(code, tree)
        if explicit != code:
            self.code = SourceLines(explicit)
            code = explicit
            tree = ast.parse(code)

        # `chat_workflow.workflow` is a legacy path with an exact, known
        # replacement. Rewriting it is mechanical; rejecting it only spends a
        # model retry on a spelling the framework can fix itself.
        canonical = _canonical_framework_imports(code)
        if canonical != code:
            self.code = SourceLines(canonical)
            code = canonical
            tree = ast.parse(code)

        joined = _join_tuple_descriptions(code, tree)
        if joined != code:
            self.code = SourceLines(joined)
            code = joined
            tree = ast.parse(code)

        # A long line is the model's most common ruff failure, and formatting
        # fixes it deterministically. Only format when it is needed, so code
        # that already fits is left exactly as written.
        if any("E501" in line for line in _ruff_violations(code)):
            formatted = _ruff_format(code)
            if formatted != code:
                self.code = SourceLines(formatted)
                code = formatted
                tree = ast.parse(code)

        fixed = _ruff_fix(code)
        if fixed != code:
            self.code = SourceLines(fixed)
            code = fixed
            tree = ast.parse(code)

        unknown = _unresolvable_imports(tree)
        if unknown:
            raise ValidationError(
                "Generated code imports module path(s) that cannot be resolved: "
                f"{', '.join(unknown)}. Import framework types from "
                "`chat_workflow` — for example "
                "`from chat_workflow import atomic_workflow_step` — and other "
                "packages from their installed names."
            )

        # A no-op validator is worse than no validator: the rule it was meant
        # to enforce is silently unenforced, and the judge has to catch it.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_model_validator(node):
                continue
            field_name = _model_validator_field_name(node)
            if field_name:
                raise ValidationError(
                    f"@model_validator('{field_name}') is not valid: "
                    "model_validator does not take a field name. Use "
                    f"@field_validator('{field_name}') for a single-field "
                    "check, or @model_validator(mode='after') for a check "
                    "across fields."
                )
            if any(isinstance(sub, ast.Pass) for sub in ast.walk(node)):
                raise ValidationError(
                    f"@model_validator '{node.name}' contains a no-op 'pass'. "
                    "A validator must enforce its rule: raise ValueError with a "
                    "specific message, or remove the validator and the "
                    "_validation_rules entry it was meant to enforce."
                )

        # Find BaseModel subclasses — try runtime issubclass first, fall back to AST
        defined_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        try:
            namespace: dict = {}
            exec(compile(tree, "<generated>", "exec"), namespace)
            from pydantic import BaseModel

            base_model_subclasses = [
                obj
                for obj in namespace.values()
                if isinstance(obj, type)
                and obj.__name__ in defined_names
                and obj is not BaseModel
                and issubclass(obj, BaseModel)
            ]
            runtime_available = True
        except Exception:
            # AST fallback: resolve which imports bring BaseModel subtypes into scope
            known_basemodel_names = _basemodel_class_names(tree)

            base_model_subclasses = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id in known_basemodel_names
                    for base in node.bases
                )
            ]
            runtime_available = False

        if len(base_model_subclasses) == 0:
            raise ValidationError("Generated code must define at least one BaseModel subclass")

        # Find public workflow steps (@atomic_workflow_step / @composite_workflow_step)
        workflow_methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            and any(
                (
                    isinstance(d, ast.Name)
                    and d.id in ("atomic_workflow_step", "composite_workflow_step")
                )
                or (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id in ("atomic_workflow_step", "composite_workflow_step")
                )
                for d in node.decorator_list
            )
        ]

        if len(workflow_methods) == 0:
            raise ValidationError(
                "Generated code must have at least one public method decorated with "
                "@atomic_workflow_step or @composite_workflow_step (imported from chat_workflow)"
            )

        # The framework supplies the session wrapper. A generated component
        # defines its steps, not a second session concept.
        if any(
            isinstance(node, ast.ClassDef) and node.name == "SessionComponent"
            for node in ast.walk(tree)
        ):
            raise ValidationError(
                "Generated code must not define a SessionComponent. The "
                "framework supplies the session wrapper; a component defines "
                "its artifact and its steps only."
            )
        for node in workflow_methods:
            is_atomic, _ = _workflow_kind(node)
            if is_atomic and node.name in ("greet", "sign_off"):
                raise ValidationError(
                    f"Generated step '{node.name}' must not define the "
                    "conversation's greeting or closing. The framework's "
                    "session wrapper owns them; name the step for the "
                    "artifact work it does."
                )

        # A chat-on docstring is a prompt. It must not restate the response
        # protocol; the schema and Instructor own it.
        for node in workflow_methods:
            is_atomic, is_chat_off = _workflow_kind(node)
            if not is_atomic or is_chat_off:
                continue
            docstring = (ast.get_docstring(node) or "").lower()
            for phrase in _PROTOCOL_PHRASES:
                if phrase in docstring:
                    raise ValidationError(
                        f"Workflow docstring for '{node.name}' states the "
                        f"response protocol ({phrase!r}). The response schema "
                        "owns the protocol; describe the conversation, not "
                        "the format."
                    )

        # One artifact: the model the workflow steps produce. Other models
        # are auxiliary — a nested value object reachable through the
        # artifact's fields, or a type a method uses or returns.
        base_names = {_class_name(cls) for cls in base_model_subclasses}
        class_nodes = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        returned = _returned_model_names(workflow_methods, base_names)
        primary_name = _resolve_primary_model(
            workflow_methods, returned, class_nodes, base_names
        )
        if primary_name is None:
            if len(base_names) > 1:
                raise ValidationError(
                    "Generated code defines multiple BaseModel subclasses "
                    f"({', '.join(sorted(base_names))}) but no workflow step "
                    "returns one. A component must have exactly one primary "
                    "artifact."
                )
            primary_name = next(iter(base_names))

        used = set(returned) | {primary_name}
        for node in workflow_methods:
            for argument in node.args.args:
                used |= _annotation_names(argument.annotation) & base_names
        for name in list(used):
            used |= _reachable_model_names(name, class_nodes, base_names)
        independent = sorted(base_names - used)
        if independent:
            raise ValidationError(
                "Generated code defines BaseModel(s) outside the primary "
                f"artifact '{primary_name}': {', '.join(independent)}. A "
                "component owns one artifact. Other models must be a field of "
                "the artifact (directly or through another nested model), or a "
                "type its methods use or return; otherwise split them into a "
                "separate component."
            )

        main_cls = next(
            cls for cls in base_model_subclasses if _class_name(cls) == primary_name
        )

        if runtime_available and isinstance(main_cls, type):
            missing_min_length = [
                name
                for name, field in main_cls.model_fields.items()
                if field.is_required()
                and field.annotation is str
                and not any(getattr(meta, "min_length", None) is not None for meta in field.metadata)
            ]
            if missing_min_length:
                raise ValidationError(
                    "Required string field(s) must declare min_length=1 "
                    "(or min_length=0 if empty is intentionally allowed): "
                    + ", ".join(missing_min_length)
                )

        for node in workflow_methods:
            is_atomic, _ = _workflow_kind(node)
            if not is_atomic:
                continue
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]  # allow a docstring
            if not (
                len(body) == 1
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and body[0].value.value is Ellipsis
            ):
                raise ValidationError(
                    f"@atomic_workflow_step method '{node.name}' must have a body of exactly '...'"
                )

        # Check that at least one method returns the class type
        main_cls_name = main_cls.__name__ if isinstance(main_cls, type) else main_cls.name
        main_cls_node = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == main_cls_name
            ),
            None,
        )
        class_docstring = ast.get_docstring(main_cls_node) if main_cls_node else None
        field_descriptions: dict[str, str] = {}
        if runtime_available and isinstance(main_cls, type):
            field_descriptions = {
                name: (field.description or "")
                for name, field in main_cls.model_fields.items()
            }
        elif main_cls_node is not None:
            for node in main_cls_node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    call = node.value
                    if not isinstance(call, ast.Call):
                        continue
                    for kw in call.keywords:
                        if (
                            kw.arg == "description"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            field_descriptions[node.target.id] = kw.value.value

        if runtime_available and isinstance(main_cls, type):
            field_names = list(main_cls.model_fields)
        elif main_cls_node is not None:
            field_names = [
                node.target.id
                for node in main_cls_node.body
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and not node.target.id.startswith("_")
                and not (
                    isinstance(node.annotation, ast.Subscript)
                    and isinstance(node.annotation.value, ast.Name)
                    and node.annotation.value.id == "ClassVar"
                )
                and not (
                    isinstance(node.annotation, ast.Name)
                    and node.annotation.id == "ClassVar"
                )
            ]
        else:
            field_names = []

        missing_descriptions = []
        tuple_descriptions = []
        for name in field_names:
            description = field_descriptions.get(name, "")
            if not isinstance(description, str):
                tuple_descriptions.append(name)
                continue
            if not description.strip():
                missing_descriptions.append(name)
        if tuple_descriptions:
            raise ValidationError(
                "Field description= must be one string, not a tuple. Wrap the "
                "text with implicit concatenation (adjacent string literals, "
                "no commas). Fields: " + ", ".join(tuple_descriptions)
            )
        if missing_descriptions:
            raise ValidationError(
                "Every field of the primary Pydantic model must pass a "
                "plain-English description= to Field(...). Missing: "
                + ", ".join(missing_descriptions)
            )

        for node in workflow_methods:
            is_atomic, is_chat_off = _workflow_kind(node)
            if not is_atomic or is_chat_off:
                # Only a chat-on atomic workflow step has a conversation prompt.
                # A chat-off or composite docstring is documentation.
                continue
            method_docstring = ast.get_docstring(node)
            if not method_docstring:
                continue
            phrase = _shared_phrase(class_docstring or "", method_docstring)
            if phrase:
                raise ValidationError(
                    f"Workflow docstring for '{node.name}' repeats the class docstring "
                    f"in the phrase {phrase!r}. The workflow docstring describes the "
                    "conversation only — what to ask, in what order, and when to stop — "
                    "not the artifact."
                )
            for field_name, description in field_descriptions.items():
                phrase = _shared_phrase(description, method_docstring)
                if phrase:
                    raise ValidationError(
                        f"Workflow docstring for '{node.name}' repeats the description "
                        f"of field '{field_name}' in the phrase {phrase!r}. Describe "
                        "the conversation, not what the model's fields mean."
                    )

        returns_class_type = any(
            (
                isinstance(n.returns, ast.Name) and n.returns.id == main_cls_name
            )
            or (
                isinstance(n.returns, ast.Constant)
                and isinstance(n.returns.value, str)
                and n.returns.value == main_cls_name
            )
            or (
                isinstance(n.returns, ast.Subscript)
                and isinstance(n.returns.value, ast.Name)
                and n.returns.value.id == "list"
                and isinstance(n.returns.slice, ast.Name)
                and n.returns.slice.id == main_cls_name
            )
            or (
                isinstance(n.returns, ast.Subscript)
                and isinstance(n.returns.value, ast.Name)
                and n.returns.value.id == "Sequence"
                and isinstance(n.returns.slice, ast.Name)
                and n.returns.slice.id == main_cls_name
            )
            for n in workflow_methods
        )
        if not returns_class_type:
            raise ValidationError(
                f"Generated code must have a public workflow step with a return "
                f"type annotation of '{main_cls_name}' — e.g. "
                f"'def generate_from_chat(cls) -> {main_cls_name}'"
            )

        # The component must enforce a non-deterministic domain rule itself.
        def _assigns_validation_rules(node: ast.AST) -> bool:
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                return False
            return any(
                isinstance(t, ast.Name) and t.id == "_validation_rules" for t in targets
            )

        has_validation_rules = any(
            _assigns_validation_rules(node)
            for node in (main_cls_node.body if main_cls_node else [])
        )

        # A private attribute is not a rule. Pydantic treats
        # `_validation_rules: list[str]` as a private attribute, so the domain
        # judge collects nothing. The correct forms are `ClassVar[list[str]]`
        # or a bare `_validation_rules = [...]`.
        for node in main_cls_node.body if main_cls_node else []:
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            if not (isinstance(target, ast.Name) and target.id == "_validation_rules"):
                continue
            annotation = node.annotation
            is_classvar = (
                isinstance(annotation, ast.Name) and annotation.id == "ClassVar"
            ) or (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and annotation.value.id == "ClassVar"
            )
            if not is_classvar:
                raise ValidationError(
                    "Declare `_validation_rules` as `ClassVar[list[str]]` (import "
                    "`ClassVar` from typing) or as a bare "
                    "`_validation_rules = [...]`. An annotated `list[str]` becomes "
                    "a Pydantic private attribute, so the rules are never "
                    "collected or judged."
                )

        uses_llm_validated = _uses_llm_validated(main_cls, tree)

        if not (has_validation_rules and uses_llm_validated):
            raise ValidationError(
                "The primary model must inherit LLMValidated and declare at "
                "least one _validation_rules entry for a non-deterministic "
                "domain business rule. Mechanical field constraints and "
                "@model_validator methods alone are not enough."
            )

        violations = _ruff_violations(code)
        if violations:
            source_lines = code.splitlines()
            details: list[str] = []
            for violation in violations[:10]:
                details.append(violation)
                match = re.match(r"generated_component\.py:(\d+):", violation)
                if match:
                    line_number = int(match.group(1))
                    if 0 < line_number <= len(source_lines):
                        details.append(
                            f"    line {line_number}: {source_lines[line_number - 1]!r}"
                        )
            raise ValidationError(
                "Generated code must pass `ruff check`. Fix these and resubmit. "
                "A line that is too long must be wrapped across lines:\n"
                + "\n".join(details)
            )

    code: SourceLines = Field(
        ...,
        description="The complete module as a JSON array of lines — one string "
        "per line, in order, with no newline escapes inside a line. "
        "It is Python source code defining the component: "
        "an LLMValidated class with one or more @atomic_workflow_step classmethods "
        "whose docstrings are good system prompts (proposes and synthesizes, "
        "includes a dialogue example), field definitions with clear business "
        "descriptions, and validation rules encoding real business semantics. "
        "The class typically has a `create` method and, when the artifact can "
        "change, an `update` method; every method works on the same artifact. "
        "The `create` conversation asks at most two or three questions and "
        "synthesizes a draft from a block of user detail rather than walking "
        "through the fields one by one. "
        "The primary model inherits LLMValidated and declares _validation_rules "
        "for the domain's non-deterministic quality criteria. Derive them from "
        "the design spec and domain knowledge; add as many as the domain "
        "needs, and never invent a rule to satisfy a checker. Cover every "
        "`what_good_looks_like` criterion with a rule or a validator. A rule "
        "is checked against the field values you define: do not write a rule "
        "that requires a field the model does not have — add the field, or "
        "leave the rule out. "
        "A _validation_rules entry is a rule the judge checks against the "
        "artifact's content, not a structural condition: a rule code can "
        "check ('notes are required when conditions were discussed') belongs "
        "in a @model_validator instead. Never write a @model_validator that "
        "only does `pass`. Declare `_validation_rules` as "
        "`ClassVar[list[str]]` (import `ClassVar`) or as a bare "
        "`_validation_rules = [...]`; never annotate it as `list[str]` — "
        "Pydantic treats that as a private attribute. "
        "The module defines one artifact — the primary BaseModel subclass its "
        "workflow steps return. Auxiliary BaseModels are allowed as nested "
        "value objects used as fields of the artifact, or as types its methods "
        "use or return; do not define a second independent artifact. "
        "The primary model's fields pass `description=` to `Field(...)`; "
        "nested models do not need descriptions. "
        "Declare date fields as `str` in ISO 8601 — never `date`/`datetime`, "
        "because the model returns JSON strings. Validate an ISO date with "
        "`datetime.date.fromisoformat(value)`, not a regex; backslashes in a "
        "regex are the most common source of invalid JSON escapes. "
        "Every @atomic_workflow_step method has a return type annotation and a body "
        "of exactly `...` (Ellipsis). Import only "
        "packages available in this environment; do not invent third-party "
        "imports. The code is valid Python and passes `ruff check`: sorted "
        "imports (a blank line between third-party and chat_workflow), no "
        "unused imports, modern type syntax (`str | None`, `list[str]`), and "
        "plain ASCII Python. Each array element is one line; do not put a "
        "newline inside an element. Escape a literal backslash as `\\\\`, use "
        "English identifiers, and never wrap the code in Markdown fences. "
        "Keep every line at most 120 characters — wrap long docstrings. "
        "Start the module with `from __future__ import annotations` so type "
        "annotations never need quotes; write `draft: MinutesDraft`, never "
        "`draft: \"MinutesDraft\"`. Use `@field_validator(\"field_name\")` for "
        "a single-field check and `@model_validator(mode=\"after\")` for a "
        "check across fields; `@model_validator` never takes a field name. "
        "A field validator is a classmethod: it takes `cls` and the value — "
        "`def _check(cls, value: str) -> str` — never `self`, and it needs no "
        "extra `@classmethod`. A model validator takes `self`: use "
        "`@model_validator(mode=\"after\")` and return `self`.",
        min_length=1,
    )

    @atomic_workflow_step(chat=False)
    @classmethod
    def generate(
        cls,
        design_spec: Annotated[
            ComponentDesignSpec,
            "Complete component design specification — domain understanding, "
            "Pydantic structure, and interaction context. The design is fully "
            "assembled; no further user input is needed.",
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 3,
    ) -> ComponentSourceCode:
        """Generate the module from the design spec: one LLMValidated class
        for the artifact, and one or more @atomic_workflow_step classmethods for
        the conversations that create and change it.

        The class docstring is one sentence naming the artifact in the user's
        language — never its fields, structure, or validation rules. Example:
        `\"\"\"Structured meeting minutes.\"\"\"`

        Give the class a `create` method, and an `update` method when the
        artifact can change after it is created. Each method's docstring is
        only about its conversation: what to ask, in what order, what to watch
        for, and when to stop. Do not describe the artifact, enumerate its
        fields, or repeat the class docstring — the model already defines
        completeness. Never reuse a field description's phrase in the
        workflow docstring; for example, if `sent_date` means "when the
        distribution email was sent", do not put that phrase in the docstring
        — describe what to ask. The agent acts as the domain_spec's
        `expert_role`; speak as that role, not as a named person. Use
        `what_good_looks_like` to decide what to probe
        for, not as text to repeat. Use details the user already gave, never
        ask for them again, confirm the draft once, then finish.
        Ask at most two or three questions in the whole conversation. When the
        user gives a block of detail, extract every field you can from it and
        propose the draft; do not walk through the fields one by one.
        Include a short dialogue example. Shape:

            from __future__ import annotations

            from chat_workflow import atomic_workflow_step

            @atomic_workflow_step
            @classmethod
            def create(cls) -> MinutesDraft:
                \"\"\"You help someone turn sketchy notes into minutes. Start
                from whatever they already told you; ask only for what is
                missing. Suggest sensible defaults where the user is unsure,
                and draw out details they only implied.

                When you have enough, show the draft once and ask if it is
                right. On confirmation, finish.

                Example:
                User: We reviewed the Q3 plan with Sarah and Mike.
                Assistant: Happy to turn that into minutes. What was decided?
                \"\"\"
                ...

            @atomic_workflow_step
            @classmethod
            def update(cls, draft: MinutesDraft) -> MinutesDraft:
                \"\"\"You help someone correct or extend existing minutes.
                Ask what should change, show the result once, and finish
                on confirmation.\"\"\"
                ...

        The design spec's domain_spec.fields and structure together define the
        model; if structure.fields is empty, derive the fields from
        domain_spec.fields. Use interaction_context for the conversation.
        `structure.base_class` is `LLMValidated`: import it from
        `chat_workflow.mixins` and inherit it. Honour `structure.model_validators`
        and `structure.extra_imports` when they are set. Required string fields
        take min_length=1 unless empty is meaningful, then min_length=0. The
        primary model's fields pass `description=` to `Field(...)`.

        The primary model must declare `_validation_rules` for the domain's
        non-deterministic quality criteria. Derive rules from the design spec
        (the user's requirements) and your own domain knowledge — you may know
        more about what makes a good artifact than the user does. Add as many
        rules as the domain genuinely needs. Every rule must be justified from
        the domain; never invent a rule to satisfy a checker. If the design spec
        and your domain knowledge do not support a rule, do not fabricate one.
        Every entry in `domain_spec.what_good_looks_like` must be represented by
        a `_validation_rules` entry or a `@model_validator`; do not drop a
        quality criterion. If it is not mechanically checkable, express it as a
        domain-quality rule.

        A rule that code can check — "notes are required when conditions were
        discussed" — belongs in a @model_validator, not in _validation_rules.
        Never write a @model_validator that only does `pass`: raise ValueError
        with a specific message, or drop the validator and the rule. Declare
        `_validation_rules: ClassVar[list[str]] = [...]` (import `ClassVar`
        from typing) or a bare `_validation_rules = [...]`; never annotate it
        as `list[str]`. Do not list a condition a field constraint or
        @model_validator already enforces.

        Nested value objects may be BaseModels, but they must be fields of the
        primary model (directly or through another nested model) or a type the
        methods use or return; never define a second independent artifact in
        the same module — example:

            class ActionItem(BaseModel):
                description: str = Field(..., description="What to do", min_length=1)
                owner: str = Field(..., description="Who owns it", min_length=1)

        Then `action_items: list[ActionItem] = Field(...)`. A dataclass or
        TypedDict is also fine for a nested shape. Declare date
        fields as `str` in ISO 8601 — never `date`/`datetime`, because the
        model returns JSON strings. Import only packages available in this
        environment; do not invent third-party imports. The module must pass
        `ruff check`: sorted imports with a blank line between third-party and
        `chat_workflow`, no unused imports, modern type syntax (`str | None`,
        `list[str]`), and every line at most 120 characters — wrap long
        docstrings. If the design cannot produce a valid module, return
        failure with the reason.
        """
        ...  # type: ignore[reportReturnType]
