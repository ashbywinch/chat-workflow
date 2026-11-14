from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, ClassVar, override

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import LLMValidated

from .design_spec import ComponentDesignSpec

_BASE_MODEL_SOURCES = frozenset({"pydantic", "chat_workflow.mixins"})


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

    line_starts = [0]
    for line in code.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        return line_starts[lineno - 1] + col

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
            insertions.append(offset(last_arg.end_lineno, last_arg.end_col_offset))

    for pos in sorted(insertions, reverse=True):
        code = code[:pos] + ", min_length=1" + code[pos:]
    return code


def _split_description(text: str, width: int) -> list[str]:
    """Split a description into lines that fit, preserving spaces at joins."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
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
                chunks = _split_description(argument.value.value, width)
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
        lines[line_number] = content[:start] + wrapped + content[end:] + newline
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
            chunks = _split_description(value.value, width)
            indent = " " * value.col_offset
            rendered = json.dumps(chunks[0])
            for chunk in chunks[1:]:
                rendered += "\n" + indent + json.dumps(chunk)
            start = sum(len(line) for line in lines[:start_line]) + value.col_offset
            end = sum(len(line) for line in lines[:end_line]) + value.end_col_offset
            replacements.append((start, end, rendered))
    for start, end, rendered in sorted(replacements, reverse=True):
        code = code[:start] + rendered + code[end:]
    return code


def _fix_field_validator_self(code: str, tree: ast.AST) -> str:
    """Rename a field validator's first argument from ``cls`` to ``self``.

    ``@field_validator`` methods are instance methods; ``cls`` trips ruff's
    N805 and confuses readers. The rename is mechanical and safe.
    """
    replacements: list[tuple[int, int, int]] = []
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
        # A classmethod validator keeps `cls`; only an instance validator
        # should take `self`.
        is_classmethod = any(
            isinstance(decorator, ast.Name) and decorator.id == "classmethod"
            for decorator in node.decorator_list
        )
        if is_classmethod:
            continue
        for argument in [*node.args.args, *node.args.kwonlyargs]:
            if argument.arg == "cls" and argument.end_col_offset is not None:
                replacements.append(
                    (argument.lineno, argument.col_offset, argument.end_col_offset)
                )
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Name)
                and child.id == "cls"
                and child.end_col_offset is not None
            ):
                replacements.append(
                    (child.lineno, child.col_offset, child.end_col_offset)
                )
    lines = code.splitlines(keepends=True)
    for lineno, start, end in sorted(replacements, reverse=True):
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:start] + "self" + line[end:]
    return "".join(lines)


def _ruff_executable() -> str | None:
    """Return the ruff binary for this environment, if present."""
    venv_ruff = Path(sys.executable).with_name("ruff")
    if venv_ruff.exists():
        return str(venv_ruff)
    return shutil.which("ruff")


def _ruff_config() -> Path | None:
    """Find the ruff config generated code is held to.

    The package lives under ``packages/<name>/src/``, so the repo root is not a
    fixed number of parents away. Ruff stops at the nearest ``pyproject.toml``
    even when it carries no ``[tool.ruff]`` section, which would silently drop
    the project's rule set — so pass the nearest config that has one.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file() and "[tool.ruff]" in candidate.read_text(encoding="utf-8"):
            return candidate
    return None


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

    config = _ruff_config()
    args = [
        ruff,
        "check",
        "--output-format=concise",
        "--stdin-filename",
        "generated_component.py",
        "-",
    ]
    if config is not None:
        args[1:1] = ["--config", str(config)]
    try:
        proc = subprocess.run(
            args,
            input=code,
            text=True,
            capture_output=True,
            cwd=config.parent if config else None,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"ruff could not run: {exc}"]

    if proc.returncode == 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _ruff_fix(code: str) -> str:
    """Apply ruff's safe automatic fixes and return the rewritten module.

    Import order, unused imports and modern type syntax are mechanical; fixing
    them here keeps model retries for judgment calls such as wrapping long
    lines. Returns the original code when ruff is unavailable.
    """
    ruff = _ruff_executable()
    if ruff is None:
        return code

    config = _ruff_config()
    args = [
        ruff,
        "check",
        "--fix",
        "--output-format=concise",
        "--stdin-filename",
        "generated_component.py",
        "-",
    ]
    if config is not None:
        args[1:1] = ["--config", str(config)]
    try:
        proc = subprocess.run(
            args,
            input=code,
            text=True,
            capture_output=True,
            cwd=config.parent if config else None,
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
        except (ImportError, ValueError):
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
            self.code = normalised
            code = normalised
            tree = ast.parse(code)

        wrapped = _wrap_long_field_calls(code, tree)
        if wrapped != code:
            self.code = wrapped
            code = wrapped
            tree = ast.parse(code)

        wrapped = _wrap_long_field_strings(code, tree)
        if wrapped != code:
            self.code = wrapped
            code = wrapped
            tree = ast.parse(code)

        fixed_self = _fix_field_validator_self(code, tree)
        if fixed_self != code:
            self.code = fixed_self
            code = fixed_self
            tree = ast.parse(code)

        # Import order, unused imports and modern syntax are mechanical; let
        # ruff fix them before any judge call.
        fixed = _ruff_fix(code)
        if fixed != code:
            self.code = fixed
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

        missing_descriptions = [
            name
            for name in field_names
            if not field_descriptions.get(name, "").strip()
        ]
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

    code: str = Field(
        ...,
        description="Complete Python source code defining the component: "
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
        "needs, and never invent a rule to satisfy a checker. "
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
        "because the model returns JSON strings. "
        "Every @atomic_workflow_step method has a return type annotation and a body "
        "of exactly `...` (Ellipsis). Import only "
        "packages available in this environment; do not invent third-party "
        "imports. The code is valid Python and passes `ruff check`: sorted "
        "imports (a blank line between third-party and chat_workflow), no "
        "unused imports, modern type syntax (`str | None`, `list[str]`), and "
        "plain ASCII Python. Use `\\n` for every newline (never a backslash "
        "followed by a space), escape a literal backslash as `\\\\`, use "
        "English identifiers, and never wrap the code in Markdown fences. "
        "Keep every line at most 120 characters — wrap long docstrings. "
        "Start the module with `from __future__ import annotations` so type "
        "annotations never need quotes; write `draft: MinutesDraft`, never "
        "`draft: \"MinutesDraft\"`. Use `@field_validator(\"field_name\")` for "
        "a single-field check and `@model_validator(mode=\"after\")` for a "
        "check across fields; `@model_validator` never takes a field name. "
        "A field validator takes `self` and the value — "
        "`def _check(self, value: str) -> str` — never `cls` and never "
        "`@classmethod`. Use `@model_validator(mode=\"after\")` and return "
        "`self`.",
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
        `expert_role`; open
        with that role. Use `what_good_looks_like` to decide what to probe
        for, not as text to repeat. Use details the user already gave, never
        ask for them again, confirm the draft once, then return success.
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
                right. On confirmation, return success.

                Example:
                User: We reviewed the Q3 plan with Sarah and Mike.
                Assistant: Happy to turn that into minutes. What was decided?
                \"\"\"
                ...

            @atomic_workflow_step
            @classmethod
            def update(cls, draft: MinutesDraft) -> MinutesDraft:
                \"\"\"You help someone correct or extend existing minutes.
                Ask what should change, show the result once, and return
                success on confirmation.\"\"\"
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
        raise NotImplementedError(  # pragma: no cover - the decorator supplies the body
            "This workflow step is implemented by its decorator"
        )
