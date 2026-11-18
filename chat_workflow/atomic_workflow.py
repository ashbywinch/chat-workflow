import copy
import json
import os
from collections.abc import Callable
from typing import Any, Generic, TypeVar

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
        self._api_base = config.api_base
        self._api_key_env = config.api_key_env
        self._max_retries = config.max_retries
        self._request_timeout_seconds = config.request_timeout_seconds
        self.response_model = config.response_model
        self._tools = config.tools
        self._tool_handlers = config.tool_handlers or {}
        self._tool_turns = config.tool_turns
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

    def _with_tools(self) -> list[dict[str, Any]]:
        """The conversation after the model has used its tools.

        Each tool call is executed by name and its result returned to the model;
        the loop stops when the model stops calling tools, or when the turn
        budget's last round asks for the answer instead (`tool_choice="none"`).
        The final call still goes through Instructor with `response_model`, so
        the result is validated and corrected exactly as a turn without tools.

        A copy is returned: the conversation history stays clean.
        """
        conversation: list[dict[str, Any]] = copy.deepcopy(self.messages)
        if not self._tools:
            return conversation
        from litellm import completion

        for turn in range(self._tool_turns):
            final = turn == self._tool_turns - 1
            answer = completion(
                model=self.model,
                messages=conversation,
                tools=self._tools,
                tool_choice="none" if final else "auto",
                api_base=self._api_base,
                api_key=os.environ.get(self._api_key_env or "OPENAI_API_KEY"),
                timeout=self._request_timeout_seconds,
            )
            message = answer.choices[0].message
            calls = list(getattr(message, "tool_calls", None) or [])
            if not calls:
                return conversation
            conversation.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                if call.function.name not in self._tool_handlers:
                    raise ValueError(f"the model called an undefined tool: {call.function.name!r}")
                result = self._tool_handlers[call.function.name](**json.loads(call.function.arguments or "{}"))
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )
        return conversation

    def _call_llm(self) -> AgentResponse[TResult]:
        from .exceptions import ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            client = get_client(
                provider=self._provider,
                api_base=self._api_base,
                api_key_env=self._api_key_env,
            )
            timer = _DebugTimer(self.debug, self.messages, self.model)

            with timer:
                # Pass a copy of messages — Instructor patches messages
                # in-place with the JSON schema. Using a copy keeps our
                # conversation history clean across turns.
                response = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                    model=self.model,
                    messages=self._with_tools(),  # pyright: ignore[reportArgumentType]
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