"""Decorators for chat-workflow: @chat and @workflow."""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel

from .conversation_orchestrator import ConversationOrchestrator
from .conversation_runtime import _get_return_type
from .debug import StreamingDebug
from .models import ConversationAction, ConversationResult
from .orchestrator_config import OrchestratorConfig
from .prompt_builder import _build_params_section, _format_docstring


def _resolve_return_type(
    raw_func: Callable[..., Any],
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> type[BaseModel]:
    """Resolve the actual return type from the function's type annotation.

    Handles Self, TypeVar, and concrete BaseModel return types.
    """
    return_type = _get_return_type(raw_func)

    if return_type is typing.Self:
        qualname = raw_func.__qualname__
        parts = qualname.rsplit(".", 1)
        if len(parts) < 2:
            raise TypeError(f"Leaf function '{func.__name__}' uses Self return type but is not a method on a class")
        class_name = parts[0]
        cls = raw_func.__globals__.get(class_name)
        if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise TypeError(
                f"Leaf function '{func.__name__}' Self return type resolves "
                f"to '{class_name}' which is not a Pydantic model subclass"
            )
        return cls

    elif isinstance(return_type, TypeVar):
        bound_type = return_type.__bound__
        if bound_type is None or not issubclass(bound_type, BaseModel):
            raise TypeError(
                f"Leaf function '{func.__name__}' TypeVar bound must be a Pydantic model, bound is {bound_type}"
            )
        actual_return_type: type[BaseModel] = return_type
        param_hints = typing.get_type_hints(raw_func)
        typevar_params = [
            name for name, annotation in param_hints.items() if name != "return" and annotation == return_type
        ]
        for param_name in typevar_params:
            if param_name in kwargs:
                actual_return_type = type(kwargs[param_name])
                break
            sig = inspect.signature(raw_func)
            param_names = list(sig.parameters.keys())
            if param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    actual_return_type = type(args[idx])
                    break
        return actual_return_type

    elif return_type is None or not issubclass(return_type, BaseModel):
        raise TypeError(
            f"Leaf function '{func.__name__}' must have a Pydantic model return type, type is {return_type}"
        )
    else:
        return return_type


def _build_system_prompt(
    raw_func: Callable[..., Any],
    kwargs: dict[str, Any],
) -> str:
    """Build the system prompt from the function's docstring and params."""
    params_section = _build_params_section(raw_func, kwargs)
    system_prompt = _format_docstring(raw_func.__doc__, **kwargs)
    if params_section:
        system_prompt += params_section

    # Guide the LLM to respect field descriptions and constraints.
    # Instructor injects the JSON schema automatically after this.
    system_prompt += (
        "\n\n## Output Format\n"
        "The JSON schema below defines the expected output. "
        "Field descriptions and constraints communicate validation rules "
        "that your response must satisfy."
    )
    return system_prompt


def _setup_orchestrator(
    system_prompt: str,
    actual_return_type: type[BaseModel],
    max_turns: int,
    debug: Any,
    tools: Any,
) -> ConversationOrchestrator:
    """Create and configure a StructuredConversationOrchestrator."""
    from .exceptions import ConversationFailedError

    return ConversationOrchestrator(
        config=OrchestratorConfig(
            system_prompt=system_prompt,
            response_model=ConversationAction[actual_return_type],
            max_turns=max_turns,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[actual_return_type].continuing(action.message or ""),
            on_success=lambda action: ConversationResult[actual_return_type].success(
                action.result,
                message="Completed successfully!",
            ),
            on_failure=lambda action: ConversationFailedError(action.message or "No reason given"),
            debug=debug,
            model=tools.config.model,
            provider=tools.config.provider,
            max_retries=tools.config.max_retries,
            request_timeout_seconds=tools.config.request_timeout_seconds,
        )
    )


def chat(func: Callable[..., Any]) -> Callable[..., Any]:
    """Auto-orchestrates an LLM chat function using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts either 'io'/'state' parameters or 'tools' (ConversationTools).
    """
    from .exceptions import ConversationFailedError

    raw_func = func.__func__ if isinstance(func, classmethod) else func

    @wraps(func)
    def wrapper(*args, **kwargs):
        tools = kwargs.pop("tools", None)
        debug = kwargs.pop("debug", None)

        if tools is None:
            raise TypeError(f"Chat function '{func.__name__}' requires 'tools' parameter")

        state = tools.state
        actual_return_type = _resolve_return_type(raw_func, func, args, kwargs)
        system_prompt = _build_system_prompt(raw_func, kwargs)
        max_turns = kwargs.pop("max_turns", 10)

        if debug is None and tools.config.debug:
            debug = StreamingDebug()

        orchestrator = _setup_orchestrator(system_prompt, actual_return_type, max_turns, debug, tools)
        result = tools.chat(orchestrator=orchestrator, first_user_input="")
        state.initial_result = result

        if result.result is None:
            raise ConversationFailedError("Conversation completed but no result was produced")

        return result.result

    return wrapper


def workflow(func: Callable[..., Any]) -> Callable[..., Any]:
    """Passes through to the wrapped function. Caller must provide ``tools=``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tools = kwargs.get("tools")
        if tools is not None:
            return func(*args, **kwargs)

        raise TypeError(f"Workflow function '{func.__name__}' requires 'tools' parameter")

    wrapper._is_workflow = True  # pyright: ignore[reportAttributeAccessIssue]  # pyright: ignore[reportAttributeAccessIssue]
    return wrapper
