import copy
from collections.abc import Callable
from typing import Generic, TypeVar

from .atomic_workflow_config import AtomicWorkflowConfig
from .debug import _DebugTimer
from .llm_interaction import ProviderType
from .models import AgentIntent, AgentResponse, TurnResult

TResult = TypeVar("TResult")


class AtomicWorkflow(Generic[TResult]):
    """Drives a single atomic workflow: one LLM conversation that produces a typed result.

    Takes an :class:`AtomicWorkflowConfig`, then :meth:`process_turn` steps
    through the conversation loop. The LLM returns an :class:`AgentResponse`
    whose ``intent`` determines what happens next (continue / success / failure).
    """

    def __init__(
        self,
        *,
        config: AtomicWorkflowConfig[TResult],
    ):
        self.messages: list[dict[str, str]] = [{"role": "system", "content": config.system_prompt}]
        self.turn_count = 0
        self.max_turns = config.max_turns
        self.model = config.model
        self._provider: ProviderType = config.provider
        self._max_retries = config.max_retries
        self._request_timeout_seconds = config.request_timeout_seconds
        self.response_model = config.response_model
        self.on_continue: Callable[[AgentResponse[TResult]], TurnResult[TResult]] = config.on_continue
        self.on_success: Callable[[AgentResponse[TResult]], TurnResult[TResult]] = config.on_success
        self.on_failure: Callable[[AgentResponse[TResult]], Exception] = config.on_failure
        self.debug = config.debug

        for message in config.initial_messages or []:
            self.messages.append(message)

    def process_turn(self, user_input: str) -> TurnResult[TResult]:
        from .exceptions import InvalidResponseError, TurnLimitExceededError

        if self.turn_count >= self.max_turns:
            raise TurnLimitExceededError(self.max_turns)

        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})

        self.turn_count += 1
        response = self._call_llm()

        message = response.message
        if message:
            self.messages.append({"role": "assistant", "content": message})

        if response.intent == AgentIntent.CONTINUE:
            return self.on_continue(response)
        if response.intent == AgentIntent.SUCCESS:
            return self.on_success(response)
        if response.intent == AgentIntent.FAILURE:
            error = self.on_failure(response)
            error.messages = list(self.messages)  # type: ignore[attr-defined]
            transcript = "".join(
                f"\n[{i}] {m.get('role', '?')}: {m.get('content', '')}" for i, m in enumerate(self.messages)
            )
            error.message = f"{error.message}\n\n━━━ CONVERSATION TRANSCRIPT ━━━{transcript}"
            if self.debug:
                self.debug.on_error(error)
            raise error

        raise InvalidResponseError(f"Invalid intent received: {response.intent}")

    def _call_llm(self) -> AgentResponse[TResult]:
        from .exceptions import ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            client = get_client(provider=self._provider)
            timer = _DebugTimer(self.debug, self.messages, self.model)

            with timer:
                # Pass a copy of messages — Instructor patches messages
                # in-place with the JSON schema. Using a copy keeps our
                # conversation history clean across turns.
                response = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                    model=self.model,
                    messages=copy.deepcopy(self.messages),  # pyright: ignore[reportArgumentType]
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