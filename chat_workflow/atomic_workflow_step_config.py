from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .debug import StreamingDebug
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
    on_failure: Callable[[AgentResponse[TResult]], Exception] | None = None
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
    # How many times to resend the same request when the provider fails
    # transiently (a 5xx, a connection drop, a rate limit). The turn is not
    # advanced, no message is appended, and the model never sees the failure:
    # a flaky gateway must not fail a conversation that was going fine.
    provider_retries: int = 3
    request_timeout_seconds: int = 30
    model_supports_tools: bool = False
    # Sent with every request; used to bypass a gateway cache.
    extra_headers: dict[str, str] | None = None
    api_base: str | None = None
    api_key_env: str | None = None
