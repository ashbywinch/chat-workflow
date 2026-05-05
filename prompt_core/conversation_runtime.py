"""Workflow-agnostic runtime helpers for interactive conversation flows."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generic, Protocol, TypeVar, Literal
from datetime import datetime

from pydantic import BaseModel, model_validator

TResult = TypeVar("TResult")


class ConversationAction(BaseModel, Generic[TResult]):
    action: Literal["continue", "success", "failure"]
    message: str | None = None
    result: TResult | None = None

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
                raise ValueError(
                    "failure action requires a message field explaining why."
                )
            if self.result is not None:
                raise ValueError("failure action cannot include result.")
        elif self.action == "success":
            if self.result is None:
                raise ValueError(
                    "success action requires a result field with the complete criteria."
                )
        return self


class ConversationResult(BaseModel, Generic[TResult]):
    result: TResult | None = None
    message: str
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> "ConversationResult[TResult]":
        return cls(result=None, message=message, is_complete=False)

    @classmethod
    def success(
        cls,
        result: TResult,
        message: str = "Completed successfully!",
    ) -> "ConversationResult[TResult]":
        return cls(result=result, message=message, is_complete=True)

    @classmethod
    def failure(cls, message: str) -> "ConversationResult[TResult]":
        return cls(result=None, message=message, is_complete=True)


class ActionLike(Protocol):
    action: str
    message: str | None


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

    def on_response(self, response: Any, duration_ms: float) -> None:
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

    def on_response(self, response: Any, duration_ms: float) -> None:
        self._print(f"{self._timestamp()}━━━ LLM RESPONSE ({duration_ms:.0f}ms) ━━━")
        try:
            if hasattr(response, "model_dump"):
                self._print(
                    f"{self._timestamp()}{json.dumps(response.model_dump(), indent=2)}"
                )
            else:
                self._print(f"{self._timestamp()}{response}")
        except Exception:
            self._print(f"{self._timestamp()}{response}")

    def on_error(self, error: Exception) -> None:
        self._print(f"{self._timestamp()}━━━ ERROR ━━━")
        self._print(f"{self._timestamp()}{type(error).__name__}: {error}")


class ConversationResultLike(Protocol):
    message: str
    is_complete: bool


class ConversationOrchestratorLike(Protocol):
    messages: list[dict[str, str]]
    turn_count: int
    model: str

    def process_turn(self, user_input: str) -> ConversationResultLike: ...


class StructuredConversationOrchestrator:
    def __init__(
        self,
        *,
        system_prompt: str,
        response_model: type[ActionLike],
        max_turns: int,
        initial_messages: list[dict[str, str]] | None,
        on_continue: Callable[[ActionLike], ConversationResultLike],
        on_success: Callable[[ActionLike], ConversationResultLike],
        on_failure: Callable[[ActionLike], Exception],
        debug: ConversationDebug | None = None,
    ):
        from .config import config

        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.turn_count = 0
        self.max_turns = max_turns
        self.model = config.model
        self.response_model = response_model
        self.on_continue = on_continue
        self.on_success = on_success
        self.on_failure = on_failure
        self.debug = debug

        for message in initial_messages or []:
            self.messages.append(message)

    def process_turn(self, user_input: str) -> ConversationResultLike:
        from .exceptions import InvalidResponseError, TurnLimitExceededError

        if self.turn_count >= self.max_turns:
            raise TurnLimitExceededError(self.max_turns)

        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})

        self.turn_count += 1
        action = self._call_llm()

        if getattr(action, "message", None):
            self.messages.append({"role": "assistant", "content": action.message})

        if action.action == "continue":
            return self.on_continue(action)
        if action.action == "success":
            return self.on_success(action)
        if action.action == "failure":
            raise self.on_failure(action)

        raise InvalidResponseError(f"Invalid action received: {action.action}")

    def _call_llm(self) -> ActionLike:
        from .config import config
        from .exceptions import ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            client = get_client(supports_tools=config.model_supports_tools)

            if self.debug:
                self.debug.on_request(self.messages, self.model)
                start = datetime.now()

            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_model=self.response_model,
                max_retries=config.max_retries,
                timeout=config.request_timeout_seconds,
            )

            if self.debug:
                duration_ms = (datetime.now() - start).total_seconds() * 1000
                self.debug.on_response(response, duration_ms)

            return response
        except ImportError as e:
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\n"
                "Install litellm for multi-provider LLM support: uv add litellm"
            )
        except Exception as e:
            if self.debug:
                self.debug.on_error(e)
            raise


@dataclass
class ConversationFlowState:
    messages: list[dict[str, str]] = field(default_factory=list)
    model: str = "unknown"
    turn_count: int = 0
    initial_result: Any = None
    final_result: Any = None


def _record_orchestrator(
    state: ConversationFlowState,
    orchestrator: ConversationOrchestratorLike,
) -> None:
    state.messages.extend(orchestrator.messages)
    state.turn_count += orchestrator.turn_count
    state.model = orchestrator.model


@dataclass
class ConversationTools:
    io: ConversationIO
    state: ConversationFlowState

    def chat(
        self,
        orchestrator: ConversationOrchestratorLike,
        first_user_input: str,
    ) -> ConversationResultLike:
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


def _get_return_type(func: Callable) -> type[BaseModel] | None:
    import typing

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

    def replace_method_call(match: re.Match) -> str:
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


def chat(func: Callable) -> Callable:
    """Auto-orchestrates an LLM chat function using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts either 'io'/'state' parameters or 'tools' (ConversationTools).
    """
    from .exceptions import ConversationFailedError

    return_type = _get_return_type(func)
    is_generic = isinstance(return_type, TypeVar)

    if is_generic:
        bound_type = return_type.__bound__
        if not issubclass(bound_type, BaseModel):
            raise TypeError(
                f"Leaf function '{func.__name__}' TypeVar bound must be a Pydantic model, bound is {bound_type}"
            )
    elif not issubclass(return_type, BaseModel):
        raise TypeError(
            f"Leaf function '{func.__name__}' must have a Pydantic model return type, type is {return_type}"
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        from .config import config

        tools = kwargs.pop("tools", None)
        io = kwargs.pop("io", None)
        state = kwargs.pop("state", None)
        debug = kwargs.pop("debug", None)

        if tools is None and io is not None:
            state = state or ConversationFlowState()
            tools = ConversationTools(io=io, state=state)

        if tools is None:
            raise TypeError(
                f"Chat function '{func.__name__}' requires 'io' or 'tools' parameter"
            )

        state = tools.state

        # Resolve TypeVar return type from actual argument
        actual_return_type = return_type
        if is_generic:
            import typing

            # Use get_type_hints() instead of inspect.signature() for annotations
            # because functions with 'from __future__ import annotations' store
            # annotations as strings, making them uncomparable to TypeVar objects.
            param_hints = typing.get_type_hints(func)
            typevar_params = [
                name
                for name, annotation in param_hints.items()
                if name != "return" and annotation == return_type
            ]
            for param_name in typevar_params:
                if param_name in kwargs:
                    actual_return_type = type(kwargs[param_name])
                    break

        # Enable debug tracing if PROMPT_CORE_DEBUG env var is set
        if debug is None and config.debug:
            debug = StreamingDebug()

        system_prompt = _format_docstring(func.__doc__, **kwargs)
        max_turns = kwargs.pop("max_turns", 10)

        orchestrator = StructuredConversationOrchestrator(
            system_prompt=system_prompt,
            response_model=ConversationAction[actual_return_type],
            max_turns=max_turns,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[
                actual_return_type
            ].continuing(action.message),
            on_success=lambda action: ConversationResult[actual_return_type].success(
                action.result,
                message="Completed successfully!",
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
            debug=debug,
        )

        result = tools.chat(orchestrator=orchestrator, first_user_input="")
        state.initial_result = result

        if result.result is None:
            raise ConversationFailedError(
                "Conversation completed but no result was produced"
            )

        return result.result

    return wrapper


def workflow(func: Callable) -> Callable:
    """Injects ConversationTools for composite functions that call other workflows."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tools = kwargs.get("tools")
        if tools is not None:
            return func(*args, **kwargs)

        io = kwargs.pop("io", None)
        if io is None:
            raise TypeError(
                f"Workflow function '{func.__name__}' requires 'io' parameter"
            )

        state = kwargs.pop("state", None) or ConversationFlowState()
        runtime_tools = ConversationTools(io=io, state=state)
        return func(*args, tools=runtime_tools, **kwargs)

    return wrapper
