"""Workflow-agnostic runtime helpers for interactive conversation flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generic, Protocol, TypeVar, Literal

from pydantic import BaseModel, model_validator

TResult = TypeVar("TResult")


class ConversationAction(BaseModel, Generic[TResult]):
    action: Literal["continue", "success", "failure"]
    message: str | None = None
    result: TResult | None = None

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and self.result is None:
            raise ValueError("success action requires result")
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
            return client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_model=self.response_model,
                max_retries=config.max_retries,
                timeout=config.request_timeout_seconds,
            )
        except ImportError as e:
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\n"
                "Install litellm for multi-provider LLM support: uv add litellm"
            )


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

    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type

    return None


def _format_docstring(docstring: str | None, **kwargs) -> str:
    if not docstring:
        return ""

    try:
        return docstring.format(**kwargs)
    except KeyError:
        return docstring


def leaf(func: Callable) -> Callable:
    """Auto-orchestrates a leaf function using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    """
    from .exceptions import ConversationFailedError

    return_type = _get_return_type(func)

    if return_type is None:
        raise TypeError(
            f"Leaf function '{func.__name__}' must have a Pydantic model return type"
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        io = kwargs.pop("io", None)
        state = kwargs.pop("state", None)

        if io is None:
            raise TypeError(f"Leaf function '{func.__name__}' requires 'io' parameter")

        if state is None:
            state = ConversationFlowState()

        tools = ConversationTools(io=io, state=state)
        system_prompt = _format_docstring(func.__doc__, **kwargs)
        max_turns = kwargs.pop("max_turns", 10)

        orchestrator = StructuredConversationOrchestrator(
            system_prompt=system_prompt,
            response_model=ConversationAction[return_type],
            max_turns=max_turns,
            initial_messages=None,
            on_continue=lambda action: ConversationResult[return_type].continuing(
                action.message
            ),
            on_success=lambda action: ConversationResult[return_type].success(
                action.result,
                message="Completed successfully!",
            ),
            on_failure=lambda action: ConversationFailedError(action.message),
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
