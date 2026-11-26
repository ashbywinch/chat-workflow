"""The one path from this library to a model.

Every model call — a workflow step's turn, the rule judge, an eval's user
simulation — is a :class:`Conversation`. It owns three things no caller should
re-implement:

* **the transcript.** PydanticAI's message list, kept here and returned on
  every turn, is the record of what the model said and was told: thinking,
  tool calls, tool returns and retry prompts included. There is no second,
  hand-built history.
* **the retry policy, one class per failure.** Transport failures are the
  provider SDK's business and are invisible. A response with no usable content
  is resent without a prompt: the model never learns it happened. A rejected
  output is a visible retry prompt carrying the validator's own complaint.
  A tool's own failure is a tool *result*, not a retry.
* **the budgets.** Each of those channels has its own count; a caller that
  judges model output as content (the review evals) sets them to zero.

A conversation is configured once and sent to repeatedly; its ``messages``
accumulate across turns.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from http import HTTPStatus
from typing import Any, ClassVar, Generic, TypeVar, cast, override

import pydantic_ai
from pydantic_ai import Agent, AgentRunResult, RetryModelRequest, ToolFailed, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from .exceptions import MaxRetriesExceededError
from .model_provider import ProviderType, build_model
from .models import AgentAnswer, FinalAnswer, as_final, as_turn

T = TypeVar("T")

# The gateway answers "I could not get a response from the provider" with an
# HTTP 400 and its own code (measured: 2005), not a 5xx. The request is sound;
# the upstream failed, so the call is transport-retried like a 503. A 400 with
# any other body is the request's fault and is raised unchanged.
_UPSTREAM_FAILURE_CODES = frozenset({2005})


def _is_upstream_failure(exc: ModelHTTPError) -> bool:
    """Whether a rejected request is really the gateway's upstream failing."""
    if exc.status_code != HTTPStatus.BAD_REQUEST:
        return False
    failure_text = "failed to get response from provider"
    body = exc.body
    entries = body if isinstance(body, list) else [body]
    for entry in entries:
        if isinstance(entry, dict):
            if entry.get("code") in _UPSTREAM_FAILURE_CODES:
                return True
            if failure_text in str(entry.get("message", "")).lower():
                return True
        elif failure_text in str(entry).lower():
            return True
    return False

# The library starts the runs, so the library owns the output: PydanticAI's
# first-run banner is for a person wiring an agent by hand, not for a workflow
# CLI or a server's logs. (The documented switch, set once before any run;
# `setattr` because the flag reads as a module constant to a checker.)
def _disable_banner() -> None:
    enabled = False
    setattr(pydantic_ai, "BANNER_ENABLED", enabled)  # noqa: B010 — a module constant to the checker


_disable_banner()

@dataclass(frozen=True)
class Resend:
    """A response that was replaced because it carried nothing usable.

    Not a turn and not a correction: it never reaches the transcript or the
    model. It is kept so a diagnosis can tell "the model said nothing twice"
    from "the model answered once, badly".
    """

    cause: str
    finish_reason: str | None


@dataclass(frozen=True)
class ModelTurn(Generic[T]):
    """One exchange with the model, validated to *T*.

    ``messages`` are the messages this run produced (the prompt, the model's
    responses, every tool call and return, every retry prompt). The
    conversation appends them to its history; a caller that stores a transcript
    stores these, not a rendering of them.
    """

    output: T
    messages: list[ModelMessage]
    usage: RunUsage
    resends: tuple[Resend, ...] = ()


class _ResendUnusable(AbstractCapability[Any]):
    """Replace a response that has no usable content, invisibly.

    A provider occasionally answers with thinking only, or stops at the token
    limit before emitting anything. There is nothing to correct and no content
    to keep, so the same request is attempted again: ``RetryModelRequest``
    stays on the same step, adds nothing to the history, and the model never
    learns that an earlier attempt happened.

    One resend is allowed per request step (the attempt number starts at 1), so
    a long tool-using turn gets the same protection on every step rather than
    spending the budget on the first.
    """

    def __init__(self, max_resends: int) -> None:
        self._max_resends = max_resends
        self.resends: list[Resend] = []

    @override
    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if self._max_resends <= 0 or request_context.attempt > self._max_resends:
            return response
        if _has_usable_content(response):
            return response
        self.resends.append(
            Resend(cause=_unusable_cause(response), finish_reason=response.finish_reason)
        )
        raise RetryModelRequest()


def _has_usable_content(response: ModelResponse) -> bool:
    """True when the response carries text or a tool call that was not cut off.

    A thinking-only response is not an answer, and a blank text part carries
    nothing a validator could accept. A response stopped at the token limit is
    not an answer either, even when text precedes the cut: the structured
    output is incomplete, so there is nothing to correct — only a fresh sample
    to take. Anything else the model wrote — even a wrong answer — is usable:
    that is the output retry's job, not this one's.
    """
    if response.finish_reason == "length":
        return False
    calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
    if calls:
        # A call whose arguments are not JSON is an encoding failure, not an
        # answer: measured as `{"response":Got it — a…}` from an endpoint
        # that treats the tool schema as a hint. Correcting it is not the
        # model's job; the same request is sampled again.
        return all(_arguments_are_object(part) for part in calls)
    text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
    # A stall emits the opening token and then whitespace to the token limit
    # (measured on the gateway: `{"response":` followed by 5,392 lines of
    # newlines). It is no answer, however long the response is.
    negligible = 0.05
    visible = len(text.strip())
    return visible > 0 and visible >= len(text) * negligible


def _arguments_are_object(part: ToolCallPart) -> bool:
    """Whether a tool call's arguments are an object, not raw model prose.

    A tool's arguments are always an object, so anything else — a list, a
    number, unparseable text — is not a call the schema could have produced.
    """
    args = part.args
    if isinstance(args, dict):
        return True
    if not isinstance(args, str):
        return True
    try:
        return isinstance(json.loads(args), dict)
    except ValueError:
        return False


def _unusable_cause(response: ModelResponse) -> str:
    if response.finish_reason == "length":
        return "token limit reached before the answer finished"
    if any(isinstance(part, ToolCallPart) and not _arguments_are_object(part) for part in response.parts):
        return "tool arguments were not JSON"
    text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
    negligible = 0.05
    if text and len(text.strip()) < len(text) * negligible:
        return "whitespace instead of an answer"
    if any(isinstance(part, TextPart) for part in response.parts):
        return "blank response"
    return "thinking only, no output"


def _failed_as_result(tool: Callable[..., Any]) -> Callable[..., Any]:
    """A tool that raises returns a failure the model can read.

    The model adapts in the same run; a raised exception would end the turn
    and tell it nothing. A tool that fails is a result, not a retry — the
    retry budget is for a call the model got wrong.
    """

    @wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Evidence for the blind catch: the contract of a step's tool is "return
        # an answer or fail"; PydanticAI turns ToolFailed into a result the model
        # reads, while any other exception ends the turn with nothing to read.
        # A caller's tool may raise anything (an SDK error, a KeyError from its
        # own data), so no narrower set can be named.
        try:
            return tool(*args, **kwargs)
        except ToolFailed:
            # a tool that raised ToolFailed itself already says what failed
            raise
        # lucidlint: ignore broad-except a step's tool may raise anything; the model reads it as a failure result
        except Exception as exc:  # noqa: BLE001 — any failure is the tool's answer
            raise ToolFailed(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


class _ProviderLog:
    """Which upstream endpoint answered, recorded for the failure message.

    The gateway names its own upstream in ``cf-aig-provider`` — on the house
    route that is always ``openrouter``, which says nothing about who actually
    answered. OpenRouter puts the serving endpoint in the response body's
    ``provider``, and its generation id in ``id`` (the id also answers at
    ``/api/v1/generation``). The endpoints behind one model differ: some enforce
    the request's tool schema, some treat it as a hint, so the log records both.
    Attaching is best-effort: a model without an httpx client records nothing.
    """

    def __init__(self) -> None:
        self.provider: str | None = None
        self.model: str | None = None
        self.generation: str | None = None

    @classmethod
    def attach(cls, model: Model) -> _ProviderLog:
        log = cls()

        async def record(response: Any) -> None:
            log.provider = response.headers.get("cf-aig-provider")
            log.model = response.headers.get("cf-aig-model")
            try:
                # The body is buffered (requests are not streamed), so reading
                # it here does not consume what the transport reads next.
                await response.aread()
                body = response.json()
            # lucidlint: ignore broad-except a provider log must never break the call
            except Exception:  # noqa: BLE001 — a log must never break a call
                return
            if isinstance(body, dict):
                log.provider = body.get("provider") or log.provider
                log.model = body.get("model") or log.model
                log.generation = body.get("id") or log.generation

        client = getattr(model, "client", None)
        http = getattr(client, "_client", None)
        hooks = getattr(http, "event_hooks", None)
        if isinstance(hooks, dict) and isinstance(hooks.get("response"), list):
            hooks["response"].append(record)
        return log

    def suffix(self) -> str:
        """`` (served by X, model Y)``, or nothing when nothing was recorded."""
        if not self.provider and not self.model:
            return ""
        parts = [part for part in (self.provider, self.model) if part]
        return f" (served by {', '.join(parts)})"


def _rejection(exc: BaseException) -> str:
    """The rejection that consumed the last retry, not only the retry count.

    ``UnexpectedModelBehavior`` is raised ``from`` the validation error that
    spent the final retry, so its cause names the fields or rules the answer
    broke. That sentence is what a caller needs to tell the model *why* the
    sample was refused; the count alone says a retry happened and nothing
    about what to do next.
    """
    cause = exc.__cause__
    text = str(cause).strip() if cause is not None else ""
    return text or str(exc)


class Conversation(Generic[T]):
    """A model, its instructions, and the transcript of one conversation.

    ``output_retries`` is the answer channel: the answer is a call to the
    ``answer`` tool (the endpoint enforces a tool's schema even when it ignores
    ``response_format``), so a rejected answer — a wrong field or a business
    rule — is fed back with the rejection and the model is asked to correct it.
    That is also the budget for a malformed call to a step's own tools.
    ``transport_retries`` is the provider SDK's resend for 429/5xx/connection
    failures — invisible, because it says nothing about the conversation.
    ``unusable_resends`` is the no-content resend above.
    ``max_requests`` bounds one turn's model steps (including tool rounds) so a
    model that never concludes fails instead of billing forever.
    """

    # The effort every reasoning request asks for. Medium is enough for this
    # work; thinking is what matters, not the level.
    REASONING_EFFORT = "medium"

    # A step may return a long generated module or a full structured result. The
    # provider default is often small enough to truncate the JSON mid-string,
    # which looks like a syntax error rather than a token limit.
    _DEFAULT_MAX_TOKENS: ClassVar[int] = 16384

    # The prompt a provider needs when the step opens the conversation: a
    # request that ends on the instructions alone is answered with an empty
    # completion.
    OPENING_PROMPT: ClassVar[str] = "Start the conversation."

    # lucidlint: ignore long-param-list the parameters are independent knobs; one production site names each
    def __init__(
        self,
        *,
        model: str,
        provider: ProviderType = "openrouter",
        instructions: str,
        output_type: Any,
        api_base: str | None = None,
        api_key_env: str | None = None,
        messages: Sequence[ModelMessage] = (),
        output_retries: int = 2,
        transport_retries: int = 2,
        unusable_resends: int = 1,
        max_requests: int = 12,
        max_tokens: int | None = None,
        timeout_seconds: float = 30.0,
        temperature: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        final_output_type: Any = None,
        tools: Sequence[Callable[..., Any]] = (),
        deps: Any = None,
        thinking: bool = False,
        model_client: Model | None = None,
    ) -> None:
        self.model = model
        self.messages: list[ModelMessage] = list(messages)
        self._provider = provider
        self._api_base = api_base
        self._api_key_env = api_key_env
        self._instructions = instructions  # lucidlint: ignore duplicate-block each line initialises a distinct field
        self._output_type = output_type
        self._output_retries = output_retries
        self._transport_retries = transport_retries
        self._unusable_resends = unusable_resends
        self._max_requests = max_requests
        self._max_tokens = max_tokens if max_tokens is not None else self._DEFAULT_MAX_TOKENS
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._extra_headers = dict(extra_headers) if extra_headers else None
        self._final_output_type = final_output_type
        self._tools = tuple(_failed_as_result(tool) for tool in tools)
        self._deps = deps
        self._provider_log = _ProviderLog()
        self._thinking = thinking
        # The injection seam for tests: a FunctionModel scripts a whole turn
        # without a network or a provider SDK. Production builds the real one
        # on the first send — constructing a step must not need a key — and
        # keeps it for the conversation's later turns.
        self._model: Model | None = model_client

    def send(
        self,
        prompt: str | None = None,
        *,
        final_turn: bool = False,
        chat_off: bool = False,
    ) -> ModelTurn[T]:
        """One turn: *prompt* (or the opening prompt), then the typed result.

        ``prompt`` may be ``None`` when the transcript already ends with a user
        turn — a step that was given its opening context as initial messages.
        ``final_turn`` tells the model this is its last chance: a step that
        would otherwise keep asking is told to answer with what it has.
        """
        if not prompt or not prompt.strip():
            has_user_turn = any(
                isinstance(message, ModelRequest)
                and any(isinstance(part, UserPromptPart) for part in message.parts)
                for message in self.messages
            )
            prompt = None if has_user_turn else self.OPENING_PROMPT

        resender = _ResendUnusable(self._unusable_resends)
        if self._model is None:
            self._model = build_model(
                provider=self._provider,
                model=self.model,
                api_base=self._api_base,
                api_key_env=self._api_key_env,
            )
        self._provider_log = _ProviderLog.attach(self._model)
        one_turn = final_turn or chat_off
        output_type = (
            (self._final_output_type or self._output_type)
            if one_turn
            else self._output_type
        )
        # A workflow step's answer is a tool call; a plain-text model call (an
        # eval's user simulation) is not a structured answer and must not be
        # forced through the tool schema.
        output_spec: Any = (
            str
            if output_type is str
            else ToolOutput(output_type, name=AgentAnswer.TOOL_NAME, strict=True)
        )
        agent: Agent[Any, T] = Agent(
            self._model,
            tools=self._tools,
            output_type=output_spec,
            system_prompt=self._instructions
            + (
                "\n\n## Final turn\n"
                "No further turn will be offered. Do not ask the user a "
                "question. Apply anything they just corrected and return "
                "intent='success' with the complete result now; return "
                "intent='failure' with the reason only if a required value "
                "cannot be produced."
                if final_turn
                else ""
            ),
            # lucidlint: ignore record-shape this is PydanticAI's retry-budget mapping, a library wire shape
            retries={"output": self._output_retries, "tools": self._output_retries},
            capabilities=[resender],
        )
        try:
            result = self._run_with_transport_retries(agent, prompt, final_turn=final_turn)
        except UnexpectedModelBehavior as exc:
            raise MaxRetriesExceededError(
                self._output_retries,
                reason=f"{_rejection(exc)}{self._provider_log.suffix()}",
            ) from exc

        output: Any = result.output
        if isinstance(output, AgentAnswer):
            output = as_turn(output)
        elif isinstance(output, FinalAnswer):
            output = as_final(output)
        turn = ModelTurn(
            output=output,
            messages=result.new_messages(),
            usage=result.usage,
            resends=tuple(resender.resends),
        )
        self.messages.extend(turn.messages)
        return turn

    def _run_with_transport_retries(
        self, agent: Agent[Any, T], prompt: str | None, *, final_turn: bool = False
    ) -> AgentRunResult[T]:
        """One turn, resent when the gateway reports its upstream failing.

        PydanticAI retries the statuses that mean "try again" (429, 5xx). The
        gateway uses a 400 with its own code for the same situation, which
        PydanticAI rightly treats as the caller's fault; this class is the one
        place that knows the exception.
        """
        attempts = self._transport_retries + 1
        for attempt in range(attempts):
            try:
                return asyncio.run(self._run_turn(agent, prompt, final_turn=final_turn))
            except ModelHTTPError as exc:
                if not _is_upstream_failure(exc) or attempt + 1 >= attempts:
                    raise
        raise AssertionError("unreachable: the last attempt raises")

    async def _run_turn(
        self, agent: Agent[Any, T], prompt: str | None, *, final_turn: bool = False
    ) -> AgentRunResult[T]:
        """One run inside its own task, with the model client opened and closed.

        ``run_sync`` starts a loop per call but never closes the client it
        opened on it; the async context manager does, so a conversation does
        not leave sockets to the garbage collector.
        """
        async with agent:
            history = list(self.messages)
            if prompt is None and not any(
                isinstance(message, ModelRequest)
                and any(isinstance(part, SystemPromptPart) for part in message.parts)
                for message in history
            ):
                # When the model is given a history and no new prompt,
                # PydanticAI does not add the agent's system prompt. The
                # instructions are part of the request, not an optional extra.
                history.insert(
                    0, ModelRequest(parts=[SystemPromptPart(self._instructions)])
                )
            return await agent.run(
                prompt,
                message_history=history,
                model_settings=self.model_settings(),
                deps=self._deps,
                usage_limits=UsageLimits(request_limit=self._max_requests),
            )

    def model_settings(self) -> ModelSettings:
        """The settings every request carries (budgets, limits, headers)."""
        # A plain dict because the reasoning effort is a provider-specific key:
        # PydanticAI's portable ``ModelSettings`` type does not name it, but the
        # OpenAI model reads it, and a dynamic gateway route needs it.
        settings: dict[str, Any] = dict(
            ModelSettings(
                max_tokens=self._max_tokens,
                timeout=self._timeout_seconds,
                max_retries=self._transport_retries,
                thinking=self._thinking,
            )
        )
        if self._temperature is not None:
            settings["temperature"] = self._temperature
        # ``thinking=True`` is the portable switch, but PydanticAI only maps it
        # to an effort for a model id its profile knows. A gateway route such as
        # ``openai/dynamic/fallback2`` is unknown, so the request would carry no
        # effort and the provider would reason not at all. Medium is enough for
        # this work; the response's reasoning is then parsed from its
        # ``reasoning``/``reasoning_content`` field.
        if self._thinking and self._provider == "openai":
            settings["openai_reasoning_effort"] = self.REASONING_EFFORT
        if self._extra_headers:
            settings["extra_headers"] = self._extra_headers
        return cast(ModelSettings, settings)
