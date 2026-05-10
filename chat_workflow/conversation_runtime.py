"""Workflow-agnostic runtime helpers for interactive conversation flows."""
from __future__ import annotations

import inspect
import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import ConversationResult
from .orchestrator import StructuredConversationOrchestrator  # noqa: F401  # re-export for backwards compat
from .protocols import ConversationIO, ConversationOrchestratorLike

TResult = TypeVar("TResult")


@dataclass
class ConversationFlowState:
    messages: list[dict[str, str]] = field(default_factory=list)
    model: str = "unknown"  # TODO: this is not type safe
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None


def _record_orchestrator[T](
    state: ConversationFlowState,
    orchestrator: ConversationOrchestratorLike[T],
) -> None:
    state.messages.extend(orchestrator.messages)
    state.turn_count += orchestrator.turn_count
    state.model = orchestrator.model


@dataclass
class ConversationTools:
    io: ConversationIO
    state: ConversationFlowState
    config: Any = None

    def chat[TResult](
        self,
        orchestrator: ConversationOrchestratorLike[TResult],
        first_user_input: str,
    ) -> ConversationResult[TResult]:
        try:
            result = orchestrator.process_turn(first_user_input)
            self.io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                user_input = self.io.prompt("\nYou")
                result = orchestrator.process_turn(user_input)
                self.io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_orchestrator(self.state, orchestrator)


def _get_return_type(func: Callable[..., Any]) -> type[BaseModel] | None:
    hints = typing.get_type_hints(func)
    return_type = hints.get("return")

    if return_type is None:
        return None

    return return_type


def _format_docstring(docstring: str | None, **kwargs) -> str:
    """Format a docstring with kwargs, supporting method calls like {var.method()}.

    Python's str.format() does NOT support method calls - {obj.method()} is
    parsed as accessing an attribute named 'method()' (with parens as part
    of the name). This function pre-processes the docstring to evaluate
    such method calls before delegating to str.format().
    """
    import re

    if not docstring:
        return ""

    # Pattern to match {identifier.method_name()} - method calls on objects
    # This handles cases like {initial_object.model_dump()}
    method_pattern = re.compile(r"\{(\w+)\.(\w+)\(\)\}")

    def replace_method_call(match: re.Match[str]) -> str:
        var_name = match.group(1)
        method_name = match.group(2)

        if var_name not in kwargs:
            # Let str.format handle missing keys (will raise KeyError)
            return match.group(0)

        obj = kwargs[var_name]
        method = getattr(obj, method_name, None)

        if method is None or not callable(method):
            # Method doesn't exist, return original placeholder
            return match.group(0)

        # Call the method and return string representation
        result = method()
        return str(result) if not isinstance(result, str) else result

    # Pre-process method calls
    processed = method_pattern.sub(replace_method_call, docstring)

    try:
        return processed.format(**kwargs)
    except KeyError:
        return processed


# Internal params that the @chat decorator consumes and should NOT
# appear in the auto-generated parameters section of the system prompt.
_INTERNAL_PARAMS: frozenset[str] = frozenset({"tools", "debug", "io", "state", "cls"})


def _get_typed_hint(func: Callable[..., Any], param_name: str) -> Any:
    """Get the resolved type hint for a parameter, preserving Annotated metadata.

    Uses typing.get_type_hints with include_extras=True so that
    Annotated[T, "description"] wrappers survive resolution,
    even through ``from __future__ import annotations``.
    """
    try:
        hints = typing.get_type_hints(func, include_extras=True)
        return hints.get(param_name)
    except Exception:
        return None


def _unwrap_annotated(hint: Any) -> Any:
    """Strip Annotated wrapper to get the bare type."""
    origin = typing.get_origin(hint)
    if origin is typing.Annotated:
        return typing.get_args(hint)[0]
    return hint


def _get_param_description(func: Callable[..., Any], param_name: str) -> str | None:
    """Extract an optional description string from Annotated[T, \"desc\"]."""
    hint = _get_typed_hint(func, param_name)
    if hint is None:
        return None
    origin = typing.get_origin(hint)
    if origin is not typing.Annotated:
        return None
    args = typing.get_args(hint)
    for arg in args[1:]:
        if isinstance(arg, str):
            return arg
    return None


def _format_type_name(hint: Any) -> str:
    """Render a type hint as a human-readable name."""
    hint = _unwrap_annotated(hint)
    if hint is None:
        return "Any"
    if isinstance(hint, TypeVar):
        return str(hint)
    if hasattr(hint, "__name__"):
        return hint.__name__
    return str(hint)


def _format_param_value(value: Any) -> str:
    """Render a parameter value for inclusion in the system prompt."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, BaseModel):
        return str(value)
    return repr(value)


def _build_params_section(func: Callable[..., Any], runtime_kwargs: dict[str, Any]) -> str:
    """Build a ``## Parameters`` section from the function signature.

    For every user-defined parameter (excluding internal plumbing like
    *tools*, *io*, *state*, *debug*) this emits:

    - Parameter name and resolved type
    - An optional description extracted from ``Annotated[T, "desc"]``
    - The runtime value when the function was called, or the default

    The section is appended to the system prompt so that workflow
    authors no longer need to manually write ``{param}`` placeholders
    in docstrings.
    """
    sig = inspect.signature(func)
    lines: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in _INTERNAL_PARAMS:
            continue

        type_name = _format_type_name(_get_typed_hint(func, param_name))
        description = _get_param_description(func, param_name)

        # Build one parameter entry
        parts = [f"- `{param_name}` ({type_name})"]
        if description:
            parts.append(f": {description}")

        # Show the effective value
        if param_name in runtime_kwargs:
            parts.append(f"\n  Value: {_format_param_value(runtime_kwargs[param_name])}")
        elif param.default is not inspect.Parameter.empty:
            parts.append(f"\n  Default: {_format_param_value(param.default)}")

        lines.append("".join(parts))

    if not lines:
        return ""
    return "\n\n## Parameters\n" + "\n".join(lines)


