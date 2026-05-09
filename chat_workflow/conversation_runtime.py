"""Workflow-agnostic runtime helpers for interactive conversation flows."""

from __future__ import annotations

import inspect
import json
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, model_validator

from .llm_interaction import ProviderType

TResult = TypeVar("TResult")


class _DebugTimer(Generic[TResult]):
    """Context manager that times an LLM call and emits debug events.

    Usage:
        timer = _DebugTimer(debug, messages, model)
        with timer:
            response = client.chat.completions.create(...)
        timer.emit_response(response)
    """

    def __init__(
        self,
        debug: ConversationDebug | None,
        messages: list[dict[str, str]],
        model: str,
    ):
        self._debug = debug
        self._messages = messages
        self._model = model
        self._start: datetime | None = None

    def __enter__(self) -> _DebugTimer[TResult]:
        if self._debug:
            self._debug.on_request(self._messages, self._model)
            self._start = datetime.now()
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def emit_response(self, response: ConversationAction[TResult]) -> None:
        if self._debug and self._start is not None:
            delta = datetime.now() - self._start
            duration_ms = delta.seconds * 1000 + delta.microseconds // 1000
            self._debug.on_response(response, duration_ms)


class ConversationAction(BaseModel, Generic[TResult]):
    action: Literal["continue", "success", "failure"]
    message: str | None = Field(
        default=None,
        description=(
            'Message for the user. Required when action is "continue" or "failure". '
            'Must be null when action is "success".'
        ),
    )
    result: TResult | None = Field(
        default=None,
        description=(
            'The criteria object. Required when action is "success". '
            'Must be null when action is "continue" or "failure".'
        ),
    )

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action == "continue":
            if not self.message:
                raise ValueError(
                    "continue action requires a message field with your question for the user. "
                    "Do not include a result field."
                )
            if self.result is not None:
                raise ValueError(
                    "continue action cannot include result. "
                    "Use action='success' if you have complete criteria to return."
                )
        elif self.action == "failure":
            if not self.message:
                raise ValueError("failure action requires a message field explaining why.")
            if self.result is not None:
                raise ValueError("failure action cannot include result.")
        elif self.action == "success":
            if self.result is None:
                raise ValueError("success action requires a result field with the complete criteria.")
        return self


class ConversationResult(BaseModel, Generic[TResult]):
    result: TResult | None = None
    message: str
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> ConversationResult[TResult]:
        return cls(result=None, message=message, is_complete=False)

    @classmethod
    def success(
        cls,
        result: TResult,
        message: str = "Completed successfully!",
    ) -> ConversationResult[TResult]:
        return cls(result=result, message=message, is_complete=True)

    @classmethod
    def failure(cls, message: str) -> ConversationResult[TResult]:
        return cls(result=None, message=message, is_complete=True)


class ConversationIO(Protocol):
    def echo(self, message: str) -> None: ...

    def prompt(self, label: str) -> str: ...


class ConversationDebug(Protocol):
    """Protocol for debugging LLM conversations.

    Implement this to receive debug events during conversation flow.
    """

    def on_request(self, messages: list[dict[str, str]], model: str) -> None:
        """Called before sending request to LLM."""
        ...

    def on_response(self, response: Any, duration_ms: int) -> None:
        """Called after receiving response from LLM."""
        ...

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        ...


class StreamingDebug:
    """A debug callback that prints LLM interactions to stdout in real-time.

    Usage:
        debug = StreamingDebug()
        orchestrator = StructuredConversationOrchestrator(..., debug=debug)
    """

    def __init__(self, file: Any = None, include_timestamps: bool = True):
        self.file = file or sys.stderr
        self.include_timestamps = include_timestamps
        self._request_start: datetime | None = None

    def _timestamp(self) -> str:
        if self.include_timestamps:
            return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
        return ""

    def _print(self, message: str) -> None:
        print(message, file=self.file, flush=True)

    def on_request(self, messages: list[dict[str, str]], model: str) -> None:
        self._request_start = datetime.now()
        self._print(f"{self._timestamp()}━━━ LLM REQUEST ━━━")
        self._print(f"{self._timestamp()}Model: {model}")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            self._print(f"{self._timestamp()}[{i}] {role}: {content}")
        self._print(f"{self._timestamp()}Waiting for response...")

    def on_response(self, response: Any, duration_ms: int) -> None:
        self._print(f"{self._timestamp()}━━━ LLM RESPONSE ({duration_ms:.0f}ms) ━━━")
        try:
            if hasattr(response, "model_dump"):
                self._print(f"{self._timestamp()}{json.dumps(response.model_dump(), indent=2)}")
            else:
                self._print(f"{self._timestamp()}{response}")
        except Exception:
            self._print(f"{self._timestamp()}{response}")

    def on_error(self, error: Exception) -> None:
        self._print(f"{self._timestamp()}━━━ ERROR ━━━")
        self._print(f"{self._timestamp()}{type(error).__name__}: {error}")


class ConversationOrchestratorLike(Protocol, Generic[TResult]):
    messages: list[dict[str, str]]
    turn_count: int
    model: str

    def process_turn(self, user_input: str) -> ConversationResult[TResult]: ...


class StructuredConversationOrchestrator(Generic[TResult]):
    def __init__(
        self,
        *,
        system_prompt: str,
        response_model: type[ConversationAction[TResult]],
        max_turns: int,
        initial_messages: list[dict[str, str]] | None,
        on_continue: Callable[[ConversationAction[TResult]], ConversationResult[TResult]],
        on_success: Callable[[ConversationAction[TResult]], ConversationResult[TResult]],
        on_failure: Callable[[ConversationAction[TResult]], Exception],
        debug: ConversationDebug | None = None,
        model: str = "default-model",
        provider: ProviderType = "openrouter",
        max_retries: int = 3,
        request_timeout_seconds: int = 30,
    ):
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.turn_count = 0
        self.max_turns = max_turns
        self.model = model
        self._provider: ProviderType = provider
        self._max_retries = max_retries
        self._request_timeout_seconds = request_timeout_seconds
        self.response_model = response_model
        self.on_continue: Callable[[ConversationAction[TResult]], ConversationResult[TResult]] = on_continue
        self.on_success: Callable[[ConversationAction[TResult]], ConversationResult[TResult]] = on_success
        self.on_failure: Callable[[ConversationAction[TResult]], Exception] = on_failure
        self.debug = debug

        for message in initial_messages or []:
            self.messages.append(message)

    def process_turn(self, user_input: str) -> ConversationResult[TResult]:
        from .exceptions import InvalidResponseError, TurnLimitExceededError

        if self.turn_count >= self.max_turns:
            raise TurnLimitExceededError(self.max_turns)

        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})

        self.turn_count += 1
        action = self._call_llm()

        message = action.message
        if message:
            self.messages.append({"role": "assistant", "content": message})

        if action.action == "continue":
            return self.on_continue(action)
        if action.action == "success":
            return self.on_success(action)
        if action.action == "failure":
            raise self.on_failure(action)

        raise InvalidResponseError(f"Invalid action received: {action.action}")

    def _call_llm(self) -> ConversationAction[TResult]:
        from .exceptions import ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            client = get_client(provider=self._provider)
            timer = _DebugTimer(self.debug, self.messages, self.model)

            with timer:
                # instructor patches the client with extra params that pyright stubs don't know about
                response = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                    model=self.model,
                    messages=self.messages,  # pyright: ignore[reportArgumentType]
                    response_model=self.response_model,
                    max_retries=self._max_retries,
                    timeout=self._request_timeout_seconds,
                )

            timer.emit_response(response)
            return response
        except ImportError as e:
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\nInstall litellm for multi-provider LLM support: uv add litellm"
            ) from e
        except Exception as e:
            if self.debug:
                self.debug.on_error(e)
            raise


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
_INTERNAL_PARAMS: frozenset[str] = frozenset({"tools", "debug", "io", "state"})


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


def chat(func: Callable[..., Any]) -> Callable[..., Any]:
    """Auto-orchestrates an LLM chat function using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts either 'io'/'state' parameters or 'tools' (ConversationTools).
    """
    from .exceptions import ConversationFailedError

    # Unwrap classmethod to get the underlying function for introspection
    raw_func = func.__func__ if isinstance(func, classmethod) else func

    @wraps(func)
    def wrapper(*args, **kwargs):
        tools = kwargs.pop("tools", None)
        debug = kwargs.pop("debug", None)

        if tools is None:
            raise TypeError(f"Chat function '{func.__name__}' requires 'tools' parameter")

        state = tools.state

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
            actual_return_type = cls

        elif isinstance(return_type, TypeVar):
            bound_type = return_type.__bound__
            if bound_type is None or not issubclass(bound_type, BaseModel):
                raise TypeError(
                    f"Leaf function '{func.__name__}' TypeVar bound must be a Pydantic model, bound is {bound_type}"
                )
            actual_return_type = return_type
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

        elif return_type is None or not issubclass(return_type, BaseModel):
            raise TypeError(
                f"Leaf function '{func.__name__}' must have a Pydantic model return type, type is {return_type}"
            )
        else:
            actual_return_type = return_type

        params_section = _build_params_section(raw_func, kwargs)
        system_prompt = _format_docstring(raw_func.__doc__, **kwargs)
        if params_section:
            system_prompt += params_section

        max_turns = kwargs.pop("max_turns", 10)

        if debug is None and tools.config.debug:
            debug = StreamingDebug()

        orchestrator = StructuredConversationOrchestrator(
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
