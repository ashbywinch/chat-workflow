"""Workflow-agnostic runtime helpers for interactive conversation flows."""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from .conversation_orchestrator import (
    ConversationOrchestrator,  # noqa: F401  # re-export for backwards compat
)

TResult = TypeVar("TResult")


def _get_return_type(func: Callable[..., Any]) -> type[BaseModel] | None:
    """Extract the return type annotation from a function.

    Uses ``typing.get_type_hints`` to resolve the ``return``
    annotation of *func*.

    Args:
        func: The function whose return type to extract.

    Returns:
        The resolved return type, or ``None`` if no return annotation
        is present.
    """
    return typing.get_type_hints(func).get("return")
