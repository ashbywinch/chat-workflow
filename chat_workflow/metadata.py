"""Introspection utilities for extracting type metadata from function signatures.

Provides helpers for resolving type hints, unwrapping ``Annotated`` wrappers,
and formatting types/values for human-readable display. Used internally by
the conversation runtime when building system prompts.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel


def _get_typed_hint(func: Callable[..., Any], param_name: str) -> Any:
    """Resolve the type hint for a parameter, preserving ``Annotated`` metadata.

    Uses ``typing.get_type_hints`` with ``include_extras=True`` so that
    ``Annotated[T, "description"]`` wrappers survive resolution even through
    ``from __future__ import annotations``.

    Args:
        func: The function whose parameter type to resolve.
        param_name: The name of the parameter to look up.

    Returns:
        The resolved type hint, or ``None`` if resolution fails or the
        parameter does not exist.
    """
    try:
        hints = typing.get_type_hints(func, include_extras=True)
        return hints.get(param_name)
    except Exception:
        return None


def _unwrap_annotated(hint: Any) -> Any:
    """Strip an ``Annotated`` wrapper to get the bare type.

    If *hint* is ``Annotated[T, ...]``, returns ``T``. Otherwise returns
    *hint* unchanged.

    Args:
        hint: A type hint, possibly wrapped in ``Annotated``.

    Returns:
        The unwrapped type.
    """
    origin = typing.get_origin(hint)
    if origin is typing.Annotated:
        return typing.get_args(hint)[0]
    return hint


def _get_param_description(func: Callable[..., Any], param_name: str) -> str | None:
    """Extract an optional description string from ``Annotated[T, "desc"]``.

    Looks for a ``str`` among the extra arguments of an ``Annotated`` type
    hint for the given parameter.

    Args:
        func: The function whose parameter to inspect.
        param_name: The name of the parameter.

    Returns:
        The description string if one is found, or ``None``.
    """
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
    """Render a type hint as a human-readable name.

    Strips any ``Annotated`` wrapper, then returns the ``__name__`` of the
    type if available.

    Args:
        hint: A type hint to format.

    Returns:
        A human-readable string representation of the type.
    """
    hint = _unwrap_annotated(hint)
    if hint is None:
        return "Any"
    if isinstance(hint, TypeVar):
        return str(hint)
    if hasattr(hint, "__name__"):
        return hint.__name__
    return str(hint)


def _format_param_value(value: Any) -> str:
    """Render a parameter value for inclusion in the system prompt.

    Strings are JSON-encoded to preserve escaping. Pydantic models are
    converted via ``str()``. Everything else uses ``repr()``.

    Args:
        value: The parameter value to format.

    Returns:
        A string representation suitable for embedding in a prompt.
    """
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, BaseModel):
        return str(value)
    return repr(value)
