"""Decorators for chat-workflow: @atomic_workflow and @composite_workflow."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .atomic_workflow import AtomicWorkflow
from .atomic_workflow_config import AtomicWorkflowConfig
from .debug import StreamingDebug
from .models import AgentResponse, TurnResult
from .prompt_builder import _build_params_section, _format_docstring
from .types_meta import resolve_return_type


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


def _setup_atomic_workflow(
    system_prompt: str,
    actual_return_type: type[BaseModel],
    max_turns: int,
    debug: Any,
    session: Any,
) -> AtomicWorkflow:
    """Create and configure an AtomicWorkflow."""
    from .exceptions import AtomicWorkflowFailedError

    return AtomicWorkflow(
        config=AtomicWorkflowConfig(
            system_prompt=system_prompt,
            response_model=AgentResponse[actual_return_type],
            max_turns=max_turns,
            initial_messages=None,
            on_continue=lambda response: TurnResult[actual_return_type].continuing(response.message or ""),
            on_success=lambda response: TurnResult[actual_return_type].success(
                response.result,
                message="Completed successfully!",
            ),
            on_failure=lambda response: AtomicWorkflowFailedError(response.message or "No reason given"),
            debug=debug,
            model=session.config.model,
            provider=session.config.provider,
            max_retries=session.config.max_retries,
            request_timeout_seconds=session.config.request_timeout_seconds,
        )
    )


def atomic_workflow(func: Callable[..., Any]) -> Callable[..., Any]:
    """Auto-orchestrates an LLM atomic workflow using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts a ``session`` parameter (:class:`~chat_workflow.session.Session`).
    """
    from .exceptions import AtomicWorkflowFailedError

    raw_func = func.__func__ if isinstance(func, classmethod) else func

    @wraps(func)
    def wrapper(*args, **kwargs):
        session = kwargs.pop("session", None)
        debug = kwargs.pop("debug", None)

        if session is None:
            raise TypeError(f"Atomic workflow '{func.__name__}' requires 'session' parameter")

        state = session.state
        actual_return_type = resolve_return_type(raw_func, func, args, kwargs)

        # Support list[X] where X is a Pydantic model
        if actual_return_type is not None:
            origin = get_origin(actual_return_type)
            if origin is list:
                inner_type = get_args(actual_return_type)[0]
                if not (isinstance(inner_type, type) and issubclass(inner_type, BaseModel)):
                    raise TypeError(
                        f"@atomic_workflow function '{func.__name__}' list return type "
                        f"resolves to list[{inner_type}] which is not a Pydantic model subclass"
                    )
                # Keep actual_return_type as list[inner_type] — valid for AgentResponse
            elif not (isinstance(actual_return_type, type) and issubclass(actual_return_type, BaseModel)):
                raise TypeError(
                    f"@atomic_workflow function '{func.__name__}' return type resolves "
                    f"to '{actual_return_type}' which is not a Pydantic model subclass"
                )

        system_prompt = _build_system_prompt(raw_func, kwargs)
        max_turns = kwargs.pop("max_turns", 10)

        if debug is None and session.config.debug:
            debug = StreamingDebug()

        workflow = _setup_atomic_workflow(system_prompt, actual_return_type, max_turns, debug, session)
        result = session.run(workflow=workflow, first_user_input="")
        state.initial_result = result

        if result.result is None:
            raise AtomicWorkflowFailedError("Atomic workflow completed but no result was produced")

        return result.result

    wrapper._is_workflow = True  # pyright: ignore[reportAttributeAccessIssue]

    return wrapper


def composite_workflow(func: Callable[..., Any]) -> Callable[..., Any]:
    """Passes through to the wrapped function. Caller must provide ``session=``.

    Handles ``@classmethod`` — if the decorated function is a classmethod
    descriptor, the underlying function is extracted, wrapped, and re-wrapped
    as a ``classmethod`` so that ``cls`` is automatically prepended on call.
    """

    raw_func = func.__func__ if isinstance(func, classmethod) else func

    @wraps(raw_func)
    def wrapper(*args, **kwargs):
        session = kwargs.get("session")
        if session is not None:
            return raw_func(*args, **kwargs)

        raise TypeError(f"Composite workflow '{raw_func.__name__}' requires 'session' parameter")

    wrapper._is_workflow = True  # pyright: ignore[reportAttributeAccessIssue]

    if isinstance(func, classmethod):
        return classmethod(wrapper)
    return wrapper
