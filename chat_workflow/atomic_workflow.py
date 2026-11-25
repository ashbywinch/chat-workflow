from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
)
from pydantic_ai.usage import RunUsage

from .atomic_workflow_config import AtomicWorkflowConfig
from .conversation import Conversation
from .exceptions import (
    AtomicWorkflowFailedError,
    InvalidResponseError,
    MaxRetriesExceededError,
    TurnLimitExceededError,
)
from .models import AgentIntent, Continue, Failure, Success, TurnResult
from .transcript import render

TResult = TypeVar("TResult")

__all__ = ["AtomicWorkflow"]

def _rejection_correction(exc: MaxRetriesExceededError) -> str:
    """The turn a rejected result earns: ask for the facts it could not hold.

    Resampling cannot fix a rule that failed for data the conversation does
    not contain — every retry answers from the same history. The rejection
    is therefore fed back as a turn, where the model can ask the user and
    the user can answer, bounded by the step's turn budget like any
    correction.
    """
    reasons = " ".join(exc.reason.split())
    limit = AtomicWorkflow._REJECTION_CORRECTION_LIMIT
    if len(reasons) > limit:
        reasons = reasons[: limit - 1].rstrip() + "…"
    return (
        "Your result was rejected by its own rules:\n"
        f"{reasons}\n\n"
        "The missing values are the user's, not yours: do not send the same "
        "result again and do not invent them. Ask for what is still missing, "
        "then return the completed result."
    )



class AtomicWorkflow(Generic[TResult]):
    """Drives a single atomic workflow: one LLM conversation that produces a typed result.

    Takes an :class:`AtomicWorkflowConfig`, then :meth:`process_turn` steps
    through the conversation loop. The LLM returns an :class:`AgentResponse`
    whose ``intent`` determines what happens next (continue / success / failure).

    The conversation — and so the transcript — belongs to a
    :class:`~chat_workflow.conversation.Conversation`; this class owns the turn
    rules (intent, turn limit) and nothing about the model.
    """

    # A rejection's own words are the rule that was broken, with its evidence
    # and a suggested repair; the turn keeps the rule and drops the bulk.
    _REJECTION_CORRECTION_LIMIT: ClassVar[int] = 2000

    # Whether this step had a user of its own. A chat-off step's turns are
    # internal; the transcript projection must not show them as conversation.
    chat: bool = True

    def __init__(
        self,
        *,
        config: AtomicWorkflowConfig[TResult],
        chat: bool = True,
    ):
        self.turn_count = 0
        self.chat = chat
        # A chat-off step has no user to ask, so it cannot need a minimum
        # number of conversational turns.
        self.min_turns = 1 if not chat else max(1, getattr(config, "min_turns", 1))
        self.max_turns = config.max_turns
        # A failure is provisional until the step has had one turn to continue.
        self._failure_given_a_turn = False
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
            thinking=True,
            final_output_type=config.final_response_model,
            tools=config.tools,
            deps=config.deps,
        )
        # The session history this step was given. Only the turns the step
        # adds are folded back into the session, so a step's prompt and its
        # initial history are not recorded twice.
        self.initial_message_count = len(self._conversation.messages)

    @property
    def messages(self) -> list[ModelMessage]:
        """The transcript: the conversation's message list, not a rendering."""
        return self._conversation.messages

    def process_turn(self, user_input: str | None = None) -> TurnResult[TResult]:
        """Take the next turn, with the corrections that turn needs.

        A correction — a rejected result, an answer that promises content it
        does not give, an early finish — is a turn of its own: it is fed back
        as the next input and spends one of ``max_turns``.
        The budget at the top of this loop is what ends a model that will not
        stop asking; no branch can add a turn past it or pass it to a callee.
        """
        while True:
            if self.turn_count >= self.max_turns:
                raise TurnLimitExceededError(self.max_turns)

            self.turn_count += 1
            # The last turn is told so: a model that would ask again is given
            # one turn to answer with what it has, and the step still fails if
            # it will not.
            final_turn = self.turn_count >= self.max_turns
            try:
                response = self._call_llm(user_input, final_turn=final_turn)
            except MaxRetriesExceededError as exc:
                # A rejection is about the result, not the sample: every retry
                # answered with what the user had already given. Resampling
                # cannot produce data the conversation does not hold, so the
                # rejection becomes a turn — the model asks for what the result
                # is missing and the user gets to answer it. Without this, a
                # draft that omits a required fact ends the run and loses the
                # conversation.
                # A chat-off step has no user to ask for the missing values;
                # the output retries already gave it the rejection.
                if final_turn or not self.chat:
                    raise
                user_input = _rejection_correction(exc)
                continue

            if response.intent == AgentIntent.CONTINUE:
                if not self.chat:
                    raise InvalidResponseError(
                        "A chat-off step returned 'continue'; its output schema "
                        "does not permit a question."
                    )
                message = (response.message or "").strip()
                if message.endswith(":") and self.turn_count < self.max_turns:
                    # A colon introduces content the turn does not hold; the
                    # model must give the list or finish, not promise it.
                    user_input = (
                        "Your message ends with a colon but the turn does not "
                        "contain the content it introduces. Give the content in "
                        "the message, or finish and let the next step show it."
                    )
                    continue
                return self.on_continue(response)
            if response.intent == AgentIntent.SUCCESS:
                if self.chat and self.turn_count < self.min_turns:
                    # The task declared that it must ask or propose first. Feed
                    # the early finish back as a correction; it is not a
                    # completion.
                    user_input = (
                        "You are finishing too early. Do the task's next step "
                        "before returning the result."
                    )
                    continue
                if (response.message or "").strip().endswith("?") and self.turn_count < self.max_turns:
                    # A success is a completion. Asking the user something
                    # while claiming the step is done leaves the answer unread.
                    user_input = (  # lucidlint: ignore duplicate-block each correction is a distinct authored prompt
                        "A success message cannot ask the user a question. If "
                        "you still need an answer, return intent='continue'. "
                        "Otherwise return the result with a statement of what "
                        "you did."
                    )
                    continue
                return self.on_success(response)
            if response.intent == AgentIntent.FAILURE:
                if not self._failure_given_a_turn and self.turn_count < self.max_turns:
                    # A step sometimes calls failure when the conversation is
                    # hard — a skeptical user, a thin description, a correction
                    # it did not want. Give it one turn to continue before
                    # accepting that the result cannot be produced.
                    self._failure_given_a_turn = True
                    user_input = (
                        "You returned failure. Continue with what you were "
                        "given: propose the result, or ask the one question you "
                        "still need. Failure is only for a user who refuses to "
                        "engage or input that cannot be interpreted."
                    )
                    continue
                error = self.on_failure(response)
                error.messages = list(self.messages)
                error.message = (
                    f"{error.message}\n\n━━━ CONVERSATION TRANSCRIPT ━━━\n"
                    f"{render(self.messages)}"
                )
                if self.debug:
                    self.debug.on_error(error)
                raise error

            raise InvalidResponseError(
                f"Invalid intent received: {response.intent}"
            )

    def _call_llm(self, user_input: str | None = None, *, final_turn: bool = False) -> Any:
        """One turn with the model, timed for the debug stream."""
        started = time.monotonic()
        if self.debug:
            self.debug.on_request(self.messages, self.model)
        turn = self._conversation.send(
            user_input, final_turn=final_turn, chat_off=not self.chat
        )
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
