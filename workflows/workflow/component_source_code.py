from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, ClassVar, override

from pydantic import BaseModel, Field
from pydantic_core import core_schema

from chat_workflow import atomic_workflow_step
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import LLMValidated

from .design_spec import ComponentDesignSpec

# lucidlint: ignore-file record-shape the repair layer exchanges AST nodes, schema fragments and tuples with libcst/ast
# lucidlint: ignore-file assembly-class each repair threads code through its helpers in one ordered pass
# lucidlint: ignore-file complexity each repair is a branch per AST case in one pass
# lucidlint: ignore-file large-function the deterministic validator is the ordered repair chain; splitting scatters it
# lucidlint: ignore-file duplicate-block the repair branches repeat apply-reparse-keep, which is the loop
# lucidlint: ignore-file swallow exec of generated code can fail arbitrarily; the AST fallback is the surface
# lucidlint: ignore-file closure-cluster the validator's repair closures are its ordered steps over code/tree


_BASE_MODEL_SOURCES = frozenset({"pydantic", "chat_workflow.mixins"})

class SourceRules:
    """The fixed facts a generated module is checked against."""

    # Phrases that restate the response protocol. The envelope and Instructor
    # own these; a generated step docstring must describe the conversation.
    PROTOCOL_PHRASES: ClassVar[tuple[str, ...]] = (
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


def _strip_markdown_fences(code: str) -> str:
    """Drop a Markdown fence wrapping the module, if the model added one."""
    text = code.strip()
    if not text.startswith("```"):
        return code
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


def _join_source_lines(value: Any) -> Any:
    """Accept the module as lines or as one string.

    A model occasionally puts several lines into one array element as
    newline escapes. The contract is one string per line, so split them
    mechanically instead of spending a retry on the escape.
    """
    if isinstance(value, list):
        return "\n".join(
            line for item in value for line in str(item).split("\n")
        )
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
        return core_schema.no_info_before_validator_function(
            _join_source_lines, core_schema.str_schema()
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
        return {"type": "array", "items": {"type": "string"}}


def _ruff_executable() -> str | None:
    """Return the ruff binary for this environment, if present."""
    venv_ruff = Path(sys.executable).with_name("ruff")
    if venv_ruff.exists():
        return str(venv_ruff)
    return shutil.which("ruff")


def _ruff_violations(code: str) -> list[str]:
    """Run ``ruff check`` on the generated module and return its diagnostics.

    Uses the repository's ``pyproject.toml`` so generated code is held to the
    same standard as the codebase. Ruff is required: a missing binary is a
    configuration error, never a silent pass.
    """
    ruff = _ruff_executable()
    if ruff is None:
        raise ValidationError(
            "ruff is required to validate generated code but was not found. "
            "Install it in the environment that generates components."
        )

    repo_root = Path.cwd()
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

    repo_root = Path.cwd()
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


def _is_plain_data_type(node: ast.ClassDef) -> bool:
    """True for an Enum, dataclass, or TypedDict — a valid field type."""
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name in {
            "Enum",
            "StrEnum",
            "IntEnum",
            "TypedDict",
            "NamedTuple",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "bytes",
        }:
            return True
    return any(
        (isinstance(d, ast.Name) and d.id == "dataclass")
        or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
        for d in node.decorator_list
    )


def _is_all_optional(node: ast.ClassDef | None) -> bool:
    """True when every public field of a model has a default."""
    if node is None:
        return False
    fields = [
        statement
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and not statement.target.id.startswith("_")
        and "ClassVar" not in _annotation_names(statement.annotation)
    ]
    return bool(fields) and all(_field_has_default(statement) for statement in fields)


def _field_has_default(statement: ast.AnnAssign) -> bool:
    """True when the annotated field has a default value."""
    value = statement.value
    if value is None:
        return False
    if isinstance(value, ast.Call):
        name = (
            value.func.id
            if isinstance(value.func, ast.Name)
            else getattr(value.func, "attr", "")
        )
        if name == "Field":
            if any(keyword.arg in {"default", "default_factory"} for keyword in value.keywords):
                return True
            if value.args:
                first = value.args[0]
                return not (
                    isinstance(first, ast.Constant) and first.value is Ellipsis
                )
            return False
    return True


def _plain_nested_types(
    node: ast.ClassDef | None, plain_names: set[str]
) -> list[str]:
    """Field annotations that name a plain class the module defines.

    ``plain_names`` are the module's class definitions that are not a
    BaseModel; a field annotation that names one of them compiles but fails
    when Pydantic builds the schema.
    """
    if node is None:
        return []
    found: set[str] = set()
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        found |= _annotation_names(statement.annotation) & plain_names
    return sorted(found)


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



def _is_field_validator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the method is decorated with ``@field_validator``."""
    return any(
        (isinstance(d, ast.Name) and d.id == "field_validator")
        or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "field_validator")
        for d in node.decorator_list
    )


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
        "For a chat-on step: REJECT if the workflow docstring restates the "
        "class docstring or enumerates the model's fields. A conversation-only "
        "docstring may name the artifact once. A chat-off step's docstring is "
        "prompt instructions and may name fields or list what to produce.",
        "In the generated model, mechanical constraints (non-empty, numeric "
        "range, pattern) use Pydantic field constraints or @model_validator. "
        "Reserve the generated model's _validation_rules for domain-quality "
        "criteria that cannot be expressed structurally — specificity, domain "
        "fit, and excellence by the user's standards.",
        "A rule must be satisfiable with what the user can reasonably supply in "
        "the conversation. REJECT a rule that demands an identifier the user "
        "would not have — a last name, an email address, an employee number — "
        "when a name they already gave identifies the person in their context.",
        "A _validation_rules entry must be decidable from the generated "
        "model's own fields. REJECT a rule that depends on a fact the artifact "
        "does not carry — evidence that a named owner attended, chronological "
        "order when there is no order or timestamp field, a decision matched to "
        "a topic when nothing links them. The fix is the field that carries the "
        "fact, or no rule; an unsatisfiable rule burns the conversation.",
        "REJECT a per-field rule that demands facts the artifact does not carry "
        "— which product, the target audience, the budget, a format — as proof "
        "of quality. A whole-artifact criterion ('someone who missed the meeting "
        "can understand it') is a rule about the artifact, not a demand on "
        "every item's description.",
        "A generated _validation_rules entry does not rest on an unstated "
        "reference. REJECT 'specific enough', 'clear', 'complete', "
        "'appropriate', or 'sufficient' unless the rule says for whom or with "
        "respect to what. 'Specific enough that someone who did not attend "
        "could act on it' names its standard; the bare word does not.",
        "REJECT a rule that turns one example from the conversation into a "
        "universal constraint — a place named 'Zoom', one date, one person's "
        "name. A rule holds for every valid artifact; the conversation's "
        "example is data, not a constraint. A rule that rejects a valid, "
        "different value (an empty action-items list, another room) is wrong.",
        "A generated step returns the result from what the user gave. A "
        "_validation_rules entry judges quality; it is never a reason to return "
        "failure when the user has nothing more to add. Return failure only "
        "when the user refuses to engage or the input cannot be interpreted.",
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
        code = _strip_markdown_fences(self.code)
        if code != self.code:
            self.code = SourceLines(code)

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

        # Formatting is deterministic and semantics-preserving; it is not a
        # lint repair. Format once, then hold the formatted code to ruff and
        # feed the actual violation back to the model through the output retry.
        formatted = _ruff_format(code)
        if formatted != code:
            self.code = SourceLines(formatted)
            code = formatted
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

        tuple_fields = sorted(
            {
                node.target.id
                for cls in ast.walk(tree)
                if isinstance(cls, ast.ClassDef)
                for node in cls.body
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and any(
                    isinstance(sub, ast.Name) and sub.id == "tuple"
                    for sub in ast.walk(node.annotation)
                )
            }
        )
        if tuple_fields:
            raise ValidationError(
                "Generated fields must not use tuple[...]: the strict answer "
                "schema rejects prefixItems. Use list[...] (a bounded list if "
                "the length is fixed): " + ", ".join(tuple_fields)
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

        # A field validator is a classmethod: it takes `cls`, not `self`.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_field_validator(node):
                continue
            args = [*node.args.args, *node.args.kwonlyargs]
            if args and args[0].arg == "self":
                raise ValidationError(
                    f"@field_validator '{node.name}' takes `self`; a field "
                    "validator is a classmethod, so it takes `cls`."
                )

        # Find BaseModel subclasses — try runtime issubclass first, fall back to AST
        defined_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        try:
            namespace: dict = {}
            exec(compile(tree, filename="<generated>", mode="exec"), namespace)
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

        # Find workflow steps (@atomic_workflow_step / @composite_workflow_step).
        # A private step is still a step: its return type is a type the
        # component uses, not a second artifact.
        all_workflow_methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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
        workflow_methods = [
            node for node in all_workflow_methods if not node.name.startswith("_")
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
            for phrase in SourceRules.PROTOCOL_PHRASES:
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
        returned = _returned_model_names(all_workflow_methods, base_names)
        primary_name = _resolve_primary_model(
            all_workflow_methods, returned, class_nodes, base_names
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
        for node in all_workflow_methods:
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

        # AST checks for the primary model's required strings and descriptions.
        # They run even when runtime introspection falls back (forward-ref
        # annotations can make Pydantic's model rebuild fail), and they raise,
        # so the model sees the error instead of the framework rewriting it.
        primary_node = class_nodes.get(primary_name)
        if primary_node is not None:
            missing_min_length: list[str] = []
            tuple_descriptions: list[str] = []
            for statement in primary_node.body:
                if not (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                ):
                    continue
                annotation = statement.annotation
                is_str = (
                    isinstance(annotation, ast.Name) and annotation.id == "str"
                ) or (
                    isinstance(annotation, ast.Constant) and annotation.value == "str"
                )
                if not is_str:
                    continue
                call = statement.value
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "Field"
                ):
                    continue
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
                if not any(kw.arg == "min_length" for kw in call.keywords):
                    missing_min_length.append(statement.target.id)
                if any(
                    kw.arg == "description" and isinstance(kw.value, ast.Tuple)
                    for kw in call.keywords
                ):
                    tuple_descriptions.append(statement.target.id)
            if tuple_descriptions:
                raise ValidationError(
                    "Field description= must be one string, not a tuple. Wrap the "
                    "text with implicit concatenation (adjacent string literals, "
                    "no commas). Fields: " + ", ".join(tuple_descriptions)
                )
            if missing_min_length:
                raise ValidationError(
                    "Required string field(s) must declare min_length=1 "
                    "(or min_length=0 if empty is intentionally allowed): "
                    + ", ".join(missing_min_length)
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

        plain_names = {
            name
            for name, node in class_nodes.items()
            if name not in base_names
            and name != primary_name
            and not _is_plain_data_type(node)
        }
        inaccessible = _plain_nested_types(
            class_nodes.get(primary_name), plain_names
        )
        if inaccessible:
            raise ValidationError(
                "Field type(s) must be Pydantic models, not plain classes: "
                + ", ".join(inaccessible)
                + ". That class must inherit LLMValidated (or BaseModel) "
                "so it can be a field of the artifact."
            )

        all_optional = [
            name
            for name in sorted(base_names - {primary_name})
            if _is_all_optional(class_nodes.get(name))
        ]
        if all_optional:
            raise ValidationError(
                "Nested model(s) must require at least one field: "
                + ", ".join(all_optional)
                + ". A model whose fields are all optional accepts an empty "
                "object, which is never an item of the artifact."
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
            main_cls_name in _annotation_names(n.returns) for n in workflow_methods
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
        "through the fields one by one. Its docstring tells the model that a "
        "success carries the complete artifact as `result` and no message, "
        "and that the final turn returns it; never return success with a "
        "question or a null result. "
        "The primary model inherits LLMValidated and declares _validation_rules "
        "for the domain's non-deterministic quality criteria. Derive them from "
        "the design spec and domain knowledge; add as many as the domain "
        "needs, and never invent a rule to satisfy a checker. Cover every "
        "`what_good_looks_like` criterion with a rule or a validator. A rule "
        "is checked against the field values you define: do not write a rule "
        "that requires a field the model does not have — add the field, or "
        "leave the rule out. A rule must be decidable from the artifact "
        "itself: if it needs a fact the artifact does not carry (whether an "
        "owner attended, whether a list is chronological without an order "
        "field, whether a covered topic has a matching decision when nothing "
        "links them), add the field or drop the rule. A rule does not rest on "
        "an unstated reference: 'specific enough' says for whom, or it is not a "
        "rule. "
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
        "regex are the most common source of invalid JSON escapes. Never use "
        "`tuple[...]` for a field; use `list[...]` (a bounded list if the "
        "length is fixed). "
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
        Tell the model in the docstring that a success carries the complete
        artifact as `result` and no message, and that the final turn returns
        it; a success with a question or a null result is wrong.
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

        The generated code must satisfy the `Generated code validation rules`
        below. That section is the checkable list; this procedure does not
        restate it. A mechanically checkable condition belongs in a field
        constraint or `@model_validator`; declare `_validation_rules` as
        `ClassVar[list[str]]` or a bare list.

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
        model returns JSON strings. Never use `tuple[...]` for a field; use
        `list[...]` (a bounded list if the length is fixed). Import only
        packages available in this
        environment; do not invent third-party imports. The module must pass
        `ruff check`: sorted imports with a blank line between third-party and
        `chat_workflow`, no unused imports, modern type syntax (`str | None`,
        `list[str]`), and every line at most 120 characters — wrap long
        docstrings. If the design cannot produce a valid module, return
        failure with the reason.
        """
        ...  # type: ignore[reportReturnType]  # the step decorator supplies the body
