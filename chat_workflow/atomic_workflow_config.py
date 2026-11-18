from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .debug import StreamingDebug
from .llm_interaction import ProviderType
from .models import AgentResponse, TurnResult

TResult = TypeVar("TResult")


@dataclass
class AtomicWorkflowConfig(Generic[TResult]):
    """Configuration for an :class:`AtomicWorkflow`."""

    system_prompt: str
    response_model: type[AgentResponse[TResult]]
    max_turns: int
    initial_messages: list[dict[str, str]] | None = None
    on_continue: Callable[[AgentResponse[TResult]], TurnResult[TResult]] | None = None
    on_success: Callable[[AgentResponse[TResult]], TurnResult[TResult]] | None = None
    on_failure: Callable[[AgentResponse[TResult]], Exception] | None = None
    debug: StreamingDebug | None = None
    model: str = "default-model"
    provider: ProviderType = "openrouter"
    api_base: str | None = None
    api_key_env: str | None = None
    max_retries: int = 3
    request_timeout_seconds: int = 30
    # The workflow's own tools: OpenAI function specs the model may call, and
    # the handlers that run them by name. A turn that calls a tool is executed
    # and fed back before the workflow asks for its result, so the model can dig
    # and still answer in `response_model`'s shape.
    tools: list[dict[str, Any]] | None = None
    tool_handlers: Mapping[str, Callable[..., Any]] | None = None
    # Tool rounds one turn may take before the workflow asks for the answer
    # (`tool_choice="none"`). Given tools and a schema, a model will otherwise
    # keep digging: measured, four searches in a row with its content beginning
    # to answer each time and never stated.
    tool_turns: int = 4