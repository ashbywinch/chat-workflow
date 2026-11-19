from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
)
from pydantic_ai.usage import RunUsage

from .atomic_workflow_config import AtomicWorkflowConfig
from .conversation import Conversation
from .exceptions import AtomicWorkflowFailedError
from .models import AgentIntent, Continue, Failure, Success, TurnResult
from .transcript import render

TResult = TypeVar("TResult")


class AtomicWorkflow(Generic[TResult]):
    """Drives a single atomic workflow: one LLM conversation that produces a typed result.

    Takes an :class:`AtomicWorkflowConfig`, then :meth:`process_turn` steps
    through the conversation loop. The LLM returns an :class:`AgentResponse`
    whose ``intent`` determines what happens next (continue / success / failure).

    The conversation — and so the transcript — belongs to a
    :class:`~chat_workflow.conversation.Conversation`; this class owns the turn
    rules (intent, turn limit) and nothing about the model.
    """

    def __init__(
        self,
        *,
        config: AtomicWorkflowConfig[TResult],
    ):
        self.turn_count = 0
        self.max_turns = config.max_turns
        self.model = config.model
        self.response_model = config.response_model
        self.on_continue: Callable[[Continue], TurnResult[TResult]] = config.on_continue
        self.on_success: Callable[[Success[TResult]], TurnResult[TResult]] = config.on_success
        self.on_failure: Callable[[Failure], AtomicWorkflowFailedError] = config.on_failure
        self.debug = config.debug
        self.usage = RunUsage()
        self._conversation = config.conversation or Conversation(
            model=config.model,
            provider=config.provider,
            instructions=self._instructions(config),
            output_type=config.response_model,
            api_base=config.api_base,
            api_key_env=config.api_key_env,
            messages=self._initial_transcript(config),
            output_retries=config.output_retries,
            transport_retries=config.transport_retries,
            unusable_resends=config.unusable_resends,
            max_requests=config.max_requests,
            max_tokens=config.max_tokens,
            timeout_seconds=config.request_timeout_seconds,
            extra_headers=config.extra_headers,
            temperature=config.temperature,
            final_output_type=config.final_response_model,
            tools=config.tools,
            deps=config.deps,
        )

    @property
    def messages(self) -> list[ModelMessage]:
        """The transcript: the conversation's message list, not a rendering."""
        return self._conversation.messages

    def process_turn(self, user_input: str) -> TurnResult[TResult]:
        from .exceptions import InvalidResponseError, TurnLimitExceededError

        if self.turn_count >= self.max_turns:
            raise TurnLimitExceededError(self.max_turns)

        self.turn_count += 1
        # The last turn is told so: a model that would ask again is given one
        # turn to answer with what it has, and the step still fails if it will not.
        final_turn = self.turn_count >= self.max_turns
        response = self._call_llm(user_input, final_turn=final_turn)

        if response.intent == AgentIntent.CONTINUE:
            return self.on_continue(response)
        if response.intent == AgentIntent.SUCCESS:
            return self.on_success(response)
        if response.intent == AgentIntent.FAILURE:
            error = self.on_failure(response)
            error.messages = list(self.messages)
            error.message = (
                f"{error.message}\n\n━━━ CONVERSATION TRANSCRIPT ━━━\n"
                f"{render(self.messages)}"
            )
            if self.debug:
                self.debug.on_error(error)
            raise error

        raise InvalidResponseError(f"Invalid intent received: {response.intent}")

    def _call_llm(self, user_input: str | None = None, *, final_turn: bool = False) -> Any:
        """One turn with the model, timed for the debug stream."""
        started = time.monotonic()
        if self.debug:
            self.debug.on_request(self.messages, self.model)
        turn = self._conversation.send(user_input, final_turn=final_turn)
        self.usage.incr(turn.usage)
        if self.debug:
            duration_ms = (time.monotonic() - started) * 1000
            self.debug.on_response(turn.output, duration_ms)
        return turn.output

    @staticmethod
    def _instructions(config: AtomicWorkflowConfig[TResult]) -> str:
        """The step's instructions, plus any system message given as initial context."""
        instructions = [config.system_prompt]
        for message in config.initial_messages or []:
            if message.get("role") == "system":
                instructions.append(message.get("content") or "")
        return "\n\n".join(part for part in instructions if part.strip())

    @staticmethod
    def _initial_transcript(config: AtomicWorkflowConfig[TResult]) -> list[ModelMessage]:
        """Initial context as transcript messages.

        The config keeps its ``role``/``content`` shape: it is the caller's
        way of seeding a step, not a second message model. A system message is
        instructions, not a turn, and is folded into the prompt above.
        """
        messages: list[ModelMessage] = []
        for message in config.initial_messages or []:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user":
                messages.append(ModelRequest.user_text_prompt(content))
            elif role == "assistant":
                messages.append(ModelResponse(parts=[TextPart(content)]))
        return messages
