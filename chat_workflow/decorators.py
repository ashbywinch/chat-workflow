"""Decorators for chat-workflow: @chat and @workflow."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from pydantic import BaseModel

from chat_workflow.types_meta import resolve_return_type

from .conversation_orchestrator import ConversationOrchestrator
from .debug import StreamingDebug
from .models import ConversationAction, ConversationResult
from .orchestrator_config import OrchestratorConfig
from .prompt_builder import _build_params_section, _format_docstring


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
    """Create and configure a ConversationOrchestrator."""
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
        actual_return_type = resolve_return_type(raw_func, func, args, kwargs)

        if actual_return_type is None or not (
            isinstance(actual_return_type, type) and issubclass(actual_return_type, BaseModel)
        ):
            raise TypeError(
                f"@chat function '{func.__name__}' return type resolves "
                f"to '{actual_return_type}' which is not a Pydantic model subclass"
            )

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
