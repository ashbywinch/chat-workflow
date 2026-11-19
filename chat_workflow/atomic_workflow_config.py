from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .conversation import Conversation
from .debug import StreamingDebug
from .exceptions import AtomicWorkflowFailedError
from .model_provider import ProviderType
from .models import Continue, Failure, Success, TurnResult

TResult = TypeVar("TResult")


@dataclass
class AtomicWorkflowConfig(Generic[TResult]):
    """Configuration for an :class:`AtomicWorkflow`.

    The budget fields are separate channels because the failures are different
    things. ``transport_retries`` resends a request the provider failed
    (429/5xx/connection); the model never sees it. ``output_retries`` feeds a
    rejected response back with its rejection, so the model can correct it.
    ``unusable_resends`` replaces a response that carried nothing (a blank or
    thinking-only completion) without telling the model. A caller that treats
    the output as content under test — an eval of the model's judgement — sets
    ``output_retries`` to 0: a malformed answer is a failure, not a draft.
    """

    system_prompt: str
    # A model or a union alias: what the request's schema is built from, so it
    # is not `type[...]` — that is the point of the field.
    response_model: Any
    max_turns: int
    # Each turn's outcome reaches the handler that asked for it, so a handler
    # reads the fields that shape actually has: a continue has a message and
    # no result, a success the reverse.
    on_continue: Callable[[Continue], TurnResult[TResult]]
    on_success: Callable[[Success[TResult]], TurnResult[TResult]]
    on_failure: Callable[[Failure], AtomicWorkflowFailedError]
    initial_messages: list[dict[str, str]] | None = None
    debug: StreamingDebug | None = None
    model: str = "default-model"
    provider: ProviderType = "openrouter"
    api_base: str | None = None
    api_key_env: str | None = None
    # The injection seam: tests pass a Conversation built on a scripted model.
    # Production leaves this None and the step builds one from the fields above.
    conversation: Conversation | None = None
    output_retries: int = 2
    transport_retries: int = 2
    unusable_resends: int = 1
    max_requests: int = 12
    max_tokens: int = 16384
    request_timeout_seconds: float = 30.0
    extra_headers: dict[str, str] | None = None
    temperature: float | None = None
    # The answer shapes used on the last allowed turn (no `continue`).
    final_response_model: Any = None
    # The step's own tools: plain functions whose signature is the schema the
    # model is given. `deps` is what a tool receives from the caller.
    tools: Sequence[Callable[..., Any]] = ()
    deps: Any = None
