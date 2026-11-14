from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .debug import StreamingDebug
from .exceptions import AtomicWorkflowStepFailedError
from .llm_interaction import ProviderType
from .models import AgentResponse, TurnResult

TResult = TypeVar("TResult")


@dataclass
class AtomicWorkflowStepConfig(Generic[TResult]):
    """Configuration for an :class:`AtomicWorkflowStep`."""

    system_prompt: str
    response_model: type[AgentResponse[TResult]]
    max_turns: int
    initial_messages: list[dict[str, str]] | None = None
    on_continue: Callable[[AgentResponse[TResult]], TurnResult[TResult]] | None = None
    on_success: Callable[[AgentResponse[TResult]], TurnResult[TResult]] | None = None
    on_failure: Callable[[AgentResponse[TResult]], AtomicWorkflowStepFailedError] | None = None
    conversation_validation_rules: dict[str, str] | None = None
    chat: bool = True
    debug: StreamingDebug | None = None
    model: str = "default-model"
    provider: ProviderType = "openrouter"
    api_base: str | None = None
    api_key_env: str | None = None
    max_retries: int = 3
    # How many times to feed a validation failure back to the model and retry
    # the call within one turn. Instructor's own retry loop does not reask for
    # arbitrary exceptions, so this is what makes deterministic and judge
    # failures visible to the model. Chat-off generation uses one sample per
    # call, so each correction is an informed retry.
    validation_corrections: int = 2
    request_timeout_seconds: float = 30
    model_supports_tools: bool = False
    api_base: str | None = None
    api_key_env: str | None = None
