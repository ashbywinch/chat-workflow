from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from chat_workflow.llm_interaction import ProviderType
from chat_workflow.models import ConversationAction, ConversationResult
from chat_workflow.protocols import ConversationDebug

TResult = TypeVar("TResult")


@dataclass
class OrchestratorConfig(Generic[TResult]):
    """Configuration for StructuredConversationOrchestrator."""

    system_prompt: str
    response_model: type[ConversationAction[TResult]]
    max_turns: int
    initial_messages: list[dict[str, str]] | None = None
    on_continue: Callable[[ConversationAction[TResult]], ConversationResult[TResult]] | None = None
    on_success: Callable[[ConversationAction[TResult]], ConversationResult[TResult]] | None = None
    on_failure: Callable[[ConversationAction[TResult]], Exception] | None = None
    debug: ConversationDebug | None = None
    model: str = "default-model"
    provider: ProviderType = "openrouter"
    max_retries: int = 3
    request_timeout_seconds: int = 30
