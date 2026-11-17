"""Build system prompt sections from function signatures and docstrings.

This module provides utilities for introspecting decorated workflow
functions and generating structured sections (parameters, formatted
docstrings) for inclusion in LLM system prompts.

It is consumed by the ``@atomic_workflow_step`` / ``@composite_workflow_step`` decorator machinery
    in :mod:`chat_workflow.decorators`.
"""

import inspect
import json
import types
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .metadata import (
    _format_param_value,
    _format_type_name,
    _get_param_description,
    _get_typed_hint,
    _unwrap_annotated,
)

# Internal params that the @atomic_workflow_step decorator consumes and should NOT
# appear in the auto-generated parameters section of the system prompt.
# ``max_turns`` is an orchestration knob, not content; the model cannot use it.
_INTERNAL_PARAMS: frozenset[str] = frozenset({"session", "debug", "cls", "max_turns"})


def _format_docstring(docstring: str | None, **kwargs) -> str:
    """Format a docstring with kwargs, supporting method calls like {var.method()}.

    Python's str.format() does NOT support method calls - {obj.method()} is
    parsed as accessing an attribute named 'method()' (with parens as part
    of the name). This function pre-processes the docstring to evaluate
    such method calls before delegating to str.format().

    Args:
        docstring: The raw docstring template, or None.
        **kwargs: Variables to interpolate into the docstring.

    Returns:
        The formatted docstring, or an empty string if docstring was
        None or empty.
    """
    import re

    if not docstring:
        return ""

    # Remove the source indentation before formatting; otherwise every prompt
    # inherits the method's 8-12 space gutter.
    docstring = inspect.cleandoc(docstring)

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


def _model_for_hint(hint: Any) -> type[BaseModel] | None:
    """Return the Pydantic model behind a parameter hint, if any.

    Handles ``Model``, ``list[Model]``, ``Model | None`` and
    ``list[Model] | None`` (including ``Annotated`` wrappers).
    """
    hint = _unwrap_annotated(hint)
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(args) == 1:
            return _model_for_hint(args[0])
        return None
    if isinstance(hint, type) and issubclass(hint, BaseModel):
        return hint
    if origin is list:
        args = typing.get_args(hint)
        if args:
            return _model_for_hint(args[0])
    return None


def input_models(func: Callable[..., Any]) -> tuple[type[BaseModel], ...]:
    """The Pydantic models behind the user-visible parameters, in order.

    Input objects are part of the prompt's vocabulary: their field names and
    descriptions reach the model through the parameter shapes, so the model
    needs the user-facing phrase for each of them for the same reason it needs
    one for the result.
    """
    models: list[type[BaseModel]] = []
    for name in inspect.signature(func).parameters:
        if name in _INTERNAL_PARAMS:
            continue
        model = _model_for_hint(_get_typed_hint(func, name))
        if model is not None and model not in models:
            models.append(model)
    return tuple(models)


def user_parameter_labels(func: Callable[..., Any]) -> dict[str, str]:
    """``{parameter name: its user-facing description}``, in signature order.

    A parameter name is internal; the author already wrote the phrase to use
    in the ``Annotated`` description, so the vocabulary section can map one to
    the other without the step prompt restating it.
    """
    labels: dict[str, str] = {}
    for name in inspect.signature(func).parameters:
        if name in _INTERNAL_PARAMS:
            continue
        description = _get_param_description(func, name)
        if description:
            labels[name] = description
    return labels


def _build_params_section(
    func: Callable[..., Any],
    runtime_kwargs: dict[str, Any],
    *,
    skip_shapes: frozenset[str] = frozenset(),
) -> str:
    """Build a ``## Parameters`` section from the function signature.

    For every user-defined parameter (excluding internal plumbing like
    *tools*, *io*, *state*, *debug*) this emits:

    - Parameter name and resolved type
    - An optional description extracted from ``Annotated[T, "desc"]``
    - The runtime value when the function was called, or the default

    The section is appended to the system prompt so that workflow
    authors no longer need to manually write ``{param}`` placeholders
    in docstrings.

    Args:
        func: The decorated workflow function to introspect.
        runtime_kwargs: The keyword arguments passed to the function
            at call time.

    Returns:
        A formatted ``## Parameters`` section string, or an empty
        string if there are no user-visible parameters.
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

        # Include the annotated shape of Pydantic model parameters. The value
        # above shows data; the schema shows the field descriptions and
        # constraints the model needs.
        model = _model_for_hint(_get_typed_hint(func, param_name))
        if model is not None and model.__name__ not in skip_shapes:
            schema = json.dumps(model.model_json_schema(), indent=2)
            parts.append(f"\n  Shape:\n```json\n{schema}\n```")

        lines.append("".join(parts))

    if not lines:
        return ""
    return "\n\n## Parameters\n" + "\n".join(lines)
