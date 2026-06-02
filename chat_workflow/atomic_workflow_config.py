from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

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
    conversation_validation_rules: list[str] | None = None
    debug: StreamingDebug | None = None
    model: str = "default-model"
    provider: ProviderType = "openrouter"
    max_retries: int = 3
    request_timeout_seconds: int = 30
