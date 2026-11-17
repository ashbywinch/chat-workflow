import copy
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, get_args, get_origin

from .atomic_workflow_step_config import AtomicWorkflowStepConfig
from .debug import _DebugTimer
from .llm_interaction import ProviderType
from .models import AgentIntent, AgentResponse, TurnResult
from .session_log import CorrectionKind, TranscriptEvent
from .transcript import Transcript

TResult = TypeVar("TResult")

# A step may return a long generated module or a full structured result. The
# provider default is often small enough to truncate the JSON mid-string, which
# looks like a syntax error rather than a token limit.
_DEFAULT_MAX_TOKENS = 16384


@dataclass(frozen=True)
class ValidationFailure:
    """One rejected model response, with the correction to send back.

    ``reason`` is the validator's message; ``guidance`` is the concrete
    remediation the framework adds for a known failure shape. ``kind`` selects
    the correction prompt: an ordinary validation retry or a language
    correction.
    """

    reason: str
    guidance: str | None = None
    rejected_text: str | None = None
    kind: CorrectionKind = "validation_correction"

    def message(self) -> str:
        if self.guidance:
            return f"{self.reason}\n\n{self.guidance}"
        return self.reason


class AtomicWorkflowStep(Generic[TResult]):
    """Drives a single atomic workflow step: one LLM conversation that produces a typed result.

    Takes an :class:`AtomicWorkflowStepConfig`, then :meth:`process_turn` steps
    through the conversation loop. The LLM returns an :class:`AgentResponse`
    whose ``intent`` determines what happens next (continue / success / failure).
    """

    def __init__(
        self,
        *,
        config: AtomicWorkflowStepConfig[TResult],
    ):
        self.messages: list[dict[str, str]] = [{"role": "system", "content": config.system_prompt}]
        self.turn_count = 0
        self.max_turns = config.max_turns
        self.model = config.model
        self._provider: ProviderType = config.provider
        self._max_retries = config.max_retries
        self._validation_corrections = config.validation_corrections
        self._provider_retries = config.provider_retries
        self._request_timeout_seconds = config.request_timeout_seconds
        self._model_supports_tools = config.model_supports_tools
        self._extra_headers = config.extra_headers or None
        self._api_base: str | None = config.api_base
        self._api_key_env: str | None = config.api_key_env
        self.chat = config.chat
        self.response_model = config.response_model
        self.on_continue: Callable[[AgentResponse[TResult]], TurnResult[TResult]] = config.on_continue
        self.on_success: Callable[[AgentResponse[TResult]], TurnResult[TResult]] = config.on_success
        self.on_failure: Callable[[AgentResponse[TResult]], Exception] = config.on_failure
        self.debug = config.debug
        # Typed audit log for this step; folded into the session log when the
        # step finishes so the transcript is structured at the source instead
        # of being inferred from message text later.
        self.events: list[TranscriptEvent] = []
        # Every validation failure the framework fed back, and whether the
        # workflow eventually produced a valid result for that turn. This is
        # how we see near-misses (recovered) instead of only hard failures.
        self.validation_events: list[dict[str, Any]] = []

        for message in config.initial_messages or []:
            self.messages.append(message)

    def process_turn(self, user_input: str) -> TurnResult[TResult]:
        from .exceptions import InvalidResponseError, TurnLimitExceededError

        if self.turn_count >= self.max_turns:
            raise TurnLimitExceededError(self.max_turns)

        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})
            self.events.append(
                TranscriptEvent(
                    type="user",
                    content=user_input,
                    turn=self.turn_count + 1,
                    audience="user",
                )
            )

        self.turn_count += 1
        response = self._call_llm()

        message = response.message
        if message and response.intent != AgentIntent.SUCCESS:
            self.messages.append({"role": "assistant", "content": message})
            self.events.append(
                TranscriptEvent(
                    type="assistant",
                    content=message,
                    turn=self.turn_count,
                    audience="user",
                )
            )

        if response.intent == AgentIntent.CONTINUE:
            return self.on_continue(response)
        if response.intent == AgentIntent.SUCCESS:
            self.events.append(
                TranscriptEvent(
                    type="result",
                    content=None,
                    turn=self.turn_count,
                    meta={"status": "success"},
                )
            )
            return self.on_success(response)
        if response.intent == AgentIntent.FAILURE:
            self.events.append(
                TranscriptEvent(
                    type="failure",
                    content=message,
                    turn=self.turn_count,
                )
            )
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
        from instructor.core.exceptions import InstructorRetryException

        from .exceptions import MaxRetriesExceededError

        client = self._client()
        for attempt in range(self._validation_corrections + 1):
            try:
                response = self._request_with_retry(client)
            except InstructorRetryException as exc:
                response, repaired_failure = _recover_from_mangled_escape(
                    exc, self.response_model
                )
                if response is None:
                    failure = repaired_failure or _validation_failure(exc)
                    if failure is None or attempt >= self._validation_corrections:
                        raise MaxRetriesExceededError(
                            self._max_retries,
                            failure.message() if failure else str(exc),
                        ) from exc
                    self._apply_correction(failure, attempt + 1)
                    continue
                self.events.append(
                    TranscriptEvent(
                        type="framework",
                        content=None,
                        turn=self.turn_count,
                        kind="repaired_escape",
                    )
                )
            except Exception as exc:
                if self.debug:
                    self.debug.on_error(exc)
                raise

            # The returned value is validated, by Pydantic and by the model's
            # own rules. The message is not: what the agent said to the user is
            # a matter of conversation quality, which the eval judges. Nothing
            # here inspects it.
            self._mark_recovered()
            return response

        raise AssertionError("unreachable")

    def _request_with_retry(self, client: Any) -> AgentResponse[TResult]:
        """Resend the same request when the provider fails transiently.

        A 5xx, a dropped connection or a rate limit says nothing about the
        conversation: the same messages are sent again, so the turn does not
        advance and the model never sees the failure. A validation error is
        not transient and is left to the correction path.
        """
        attempts = max(1, self._provider_retries)
        result_fields = _result_field_names(self.response_model)
        for attempt in range(attempts):
            try:
                return self._request(client)
            except Exception as exc:
                retryable = (
                    _is_transient_provider_error(exc)
                    or _returned_no_content(exc)
                    or _returned_no_envelope(exc, result_fields)
                )
                if attempt + 1 >= attempts or not retryable:
                    raise
                self.events.append(
                    TranscriptEvent(
                        type="framework",
                        content=None,
                        turn=self.turn_count,
                        kind="provider_retry",
                        meta={"attempt": attempt + 1, "error": type(exc).__name__},
                    )
                )
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")

    def _client(self) -> Any:
        """Build the provider client, translating a missing provider cleanly."""
        from .exceptions import ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            return get_client(
                provider=self._provider,
                api_key_env=self._api_key_env,
                api_base=self._api_base,
                model_supports_tools=self._model_supports_tools,
            )
        except ImportError as exc:
            raise ProviderNotFoundError(
                f"No LLM providers available. {exc}\nInstall litellm for "
                "multi-provider LLM support: uv add litellm"
            ) from exc
        except Exception as exc:
            if self.debug:
                self.debug.on_error(exc)
            raise

    def _request(self, client: Any) -> AgentResponse[TResult]:
        """One model sample, with the framework owning the correction loop.

        Letting Instructor retry blind would repeat the same messages; a
        gateway may even answer an identical prompt from cache, so the model
        never sees the error. One sample per call makes every correction an
        informed retry.
        """
        timer = _DebugTimer(self.debug, self.messages, self.model)
        instructor_retries = (
            1 if self._validation_corrections > 0 else self._max_retries
        )
        messages = copy.deepcopy(self.messages)
        if not _has_cjk(_language_reference(self.messages)):
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "The conversation is in English. Reply in English.",
                },
            )
        messages = _request_messages(messages)
        with timer:
            # Pass a copy: Instructor patches messages in place with the JSON
            # schema, and the conversation history must stay clean.
            response = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                model=self.model,
                messages=messages,  # pyright: ignore[reportArgumentType]
                response_model=self.response_model,
                max_retries=instructor_retries,
                timeout=self._request_timeout_seconds,
                max_tokens=_DEFAULT_MAX_TOKENS,
                extra_headers=self._extra_headers,
            )
        timer.emit_response(response)
        return response

    def _apply_correction(self, failure: ValidationFailure, attempt: int) -> None:
        """Feed one rejected response and its correction back to the model."""
        self.validation_events.append({
            "turn": self.turn_count,
            "attempt": attempt,
            "reason": failure.message(),
            "recovered": False,
        })
        if failure.rejected_text:
            self.messages.append(
                {"role": "assistant", "content": failure.rejected_text}
            )
            self.events.append(
                TranscriptEvent(
                    type="framework",
                    content=failure.rejected_text,
                    turn=self.turn_count,
                    kind="rejected_response",
                )
            )
        correction = _correction_prompt(failure)
        # The correction is the next turn in the exchange. It carries the
        # `[framework]` prefix, so it is never mistaken for something the user
        # said, and the request still ends on a user turn.
        self.messages.append({"role": "user", "content": correction})
        self.events.append(
            TranscriptEvent(
                type="framework",
                content=correction,
                turn=self.turn_count,
                kind=failure.kind,
            )
        )

    def _mark_recovered(self) -> None:
        for event in self.validation_events:
            if event["turn"] == self.turn_count:
                event["recovered"] = True


def _language_reference(messages: list[dict[str, Any]]) -> str:
    """The text that fixes the conversation's language.

    The last real user turn wins; before the user has spoken, the system
    prompt does. Framework corrections are English boilerplate and are never
    the reference.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content") or ""
            if not content.lstrip().startswith("[framework]"):
                return content
    for message in messages:
        if message.get("role") == "system":
            return message.get("content") or ""
    return ""


def _has_cjk(text: str) -> bool:
    """True when the text contains any CJK character."""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _inner_result_model(response_model: object) -> object:
    """Unwrap ``AgentResponse[list[Model]]`` to the model that declares labels."""
    model = response_model
    generic = getattr(model, "__pydantic_generic_metadata__", None)
    args = (generic or {}).get("args") or get_args(model)
    if args:
        model = args[0]
    if get_origin(model) is list:
        inner = get_args(model)
        if inner:
            model = inner[0]
    return model


def _result_field_names(response_model: object) -> frozenset[str]:
    """The result model's field names, for telling a domain object from a non-answer."""
    fields = getattr(_inner_result_model(response_model), "model_fields", None)
    return frozenset(fields) if fields else frozenset()


def _user_text(messages: list[dict[str, Any]]) -> str:
    """Everything the user has actually said, excluding framework turns."""
    return " ".join(
        message.get("content") or ""
        for message in messages
        if message.get("role") == "user"
        and not (message.get("content") or "").lstrip().startswith("[framework]")
    )



def _mentions(text: str, term: str) -> bool:
    """Match a term across spaces, underscores, and hyphens, on word boundaries."""
    parts = [part for part in re.split(r"[\s_-]+", term) if part]
    if not parts:
        return False
    pattern = (
        r"(?<![A-Za-z0-9_])"
        + r"[\s_-]+".join(re.escape(part) for part in parts)
        + r"(?![A-Za-z0-9_])"
    )
    return re.search(pattern, text, re.IGNORECASE) is not None


_FAILURE_GUIDANCE: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("invalid escape",),
        " The JSON string has an invalid escape. Use `\\n` for every newline "
        "(never a backslash followed by a space), escape a literal backslash "
        "as `\\\\`, and do not wrap code in Markdown.",
    ),
    (
        ("invalid decimal literal", "expected an indented block"),
        " Write plain ASCII Python: remove Markdown fences, full-width "
        "characters, and non-ASCII punctuation.",
    ),
    (
        ("invalid syntax", "unexpected EOF", "was never closed"),
        " Check the offending line. If the code ends mid-statement, the "
        "response was cut off; return a shorter complete module.",
    ),
    (
        ("expected value at line 1 column 1",),
        " You replied with plain text instead of JSON. Return only a JSON "
        'object: {"intent": "continue", "message": "<your reply to the user>", '
        '"result": null} or {"intent": "success", "message": null, '
        '"result": {...}}. Put the user-facing text in `message`.',
    ),
    (
        ("control character", "EOF while parsing", "Invalid JSON"),
        " Return valid JSON only. Encode every newline inside a string as "
        "\\n, never a literal line break; escape every double quote inside a "
        'string as \\"; close every brace and bracket.',
    ),
)


def _request_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Shape the transcript for the wire, or raise on an invalid one.

    The rules and their sources live in :class:`chat_workflow.transcript.Transcript`.
    This is the single place the model-facing request is assembled, so a
    malformed history fails here rather than as an empty completion later.
    """
    transcript = Transcript.from_messages(messages)
    if not transcript.turns:
        # The step opens the conversation; a chat request needs a turn to
        # answer, and a system-only request comes back empty.
        transcript.user("Start the conversation.")
    return transcript.wire()


def _describe_model_errors(error: Any) -> str:
    """The model's broken rules, as the rules themselves state them.

    Pydantic's ``str()`` buries the rule inside a truncated copy of the whole
    input followed by a documentation link, and labels a model-level failure
    with the envelope name rather than the model that rejected it. The model
    needs two things: which rule it broke, and where that rule applies.
    """
    lines = []
    for detail in error.errors():
        where = ".".join(
            str(part) for part in detail.get("loc", ()) if part != "result"
        )
        rule = _rule_text(str(detail.get("msg", "")).strip())
        lines.append(f"- {where}: {rule}" if where else f"- {rule}")

    name = _result_model_name(str(getattr(error, "title", "") or "the result"))
    count = len(lines)
    if count == 0:
        return f"{name} rejected this result"
    if count == 1:
        return f"{name} rejected this result:\n{lines[0]}"
    return f"{name} rejected this result on {count} rules:\n" + "\n".join(lines)


def _result_model_name(title: str) -> str:
    """`AgentResponse[MinutesDraft]` is the envelope; the model is `MinutesDraft`."""
    match = re.search(r"\[(.+)\]$", title)
    return match.group(1) if match else title


def _rule_text(message: str) -> str:
    """Drop Pydantic's error-type prefix, which carries no information."""
    for prefix in ("Value error, ", "Assertion failed, "):
        if message.startswith(prefix):
            return message[len(prefix):]
    return message


def _validation_failure(exc: Exception) -> ValidationFailure | None:
    """Turn Instructor's last failed attempt into a correction, if it is one."""
    from json import JSONDecodeError

    from pydantic import ValidationError as PydanticValidationError

    from .exceptions import ValidationError

    attempts = getattr(exc, "failed_attempts", None) or []
    if not attempts:
        return None
    cause = getattr(attempts[-1], "exception", None)
    if not isinstance(cause, (PydanticValidationError, ValidationError, JSONDecodeError)):
        return None
    reason = (
        _describe_model_errors(cause)
        if isinstance(cause, PydanticValidationError)
        else str(cause).strip()
    )
    guidance = next(
        (
            text
            for needles, text in _FAILURE_GUIDANCE
            if any(needle in reason for needle in needles)
        ),
        None,
    )
    rejected_text = _completion_text(exc)
    if rejected_text is not None and not rejected_text.lstrip().startswith("{"):
        # A plain-text reply is not a turn in this protocol. Echoing it back
        # as an assistant message invites the model to keep chatting instead
        # of returning the envelope.
        rejected_text = None
    return ValidationFailure(
        reason=reason,
        guidance=guidance,
        rejected_text=rejected_text,
    )


def _correction_prompt(failure: ValidationFailure) -> str:
    """The framework turn that tells the model how to repair its response."""
    return (
        "[framework] Your previous response was rejected by validation. "
        f"Fix exactly these violations:\n\n{failure.message()}\n\n"
        "Keep everything that was valid; change only what the violations "
        "name. Return the complete structured response again as JSON only: "
        '{"intent": "continue", "message": "<your reply to the user>", '
        '"result": null} or {"intent": "success", "message": null, '
        '"result": {...}}. Put the user-facing text in `message`, never as '
        "plain text. Do not reply with a description of the violations."
    )


def _completion_object(exc: Exception) -> Any:
    """The completion that was rejected, if the provider returned one."""
    completion = getattr(exc, "last_completion", None)
    if completion is None:
        attempts = getattr(exc, "failed_attempts", None) or []
        if attempts:
            completion = getattr(attempts[-1], "completion", None)
    return completion


def _completion_text(exc: Exception) -> str | None:
    """Return the assistant content from the last rejected completion."""
    return _message_content(_completion_object(exc))


def _failed_attempt_text(exc: Exception) -> str | None:
    """The content of the attempt that actually failed to parse."""
    attempts = getattr(exc, "failed_attempts", None) or []
    if attempts:
        text = _message_content(getattr(attempts[-1], "completion", None))
        if text is not None:
            return text
    return _completion_text(exc)


def _message_content(completion: Any) -> str | None:
    """The assistant text inside one chat completion, if there is any."""
    choices = getattr(completion, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return None


_JSON_ESCAPES = '"\\/bfnrt'


def _repair_json_escapes(text: str) -> str:
    """Repair the escape corruption a provider introduces in long strings.

    Inside a long string (a Mermaid diagram, a generated module) the provider
    sometimes drops the character after a backslash: ``\\n`` arrives as ``\\1``
    or as a bare backslash followed by the line's indentation. Neither is a
    valid JSON escape, so a complete response fails to parse.

    A backslash before a digit or whitespace is always that mangling. Any
    other stray backslash is a literal the model forgot to escape — a regex
    ``\\d`` — so it is doubled instead, which preserves it.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        following = text[index + 1] if index + 1 < length else ""
        if following in _JSON_ESCAPES:
            out.append(char)
            out.append(following)
            index += 2
            continue
        if following == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", text[index + 2 : index + 6]):
            out.append(text[index : index + 6])
            index += 6
            continue
        if following.isspace():
            # The backslash replaced the `n`; the whitespace after it is the
            # line's own indentation and must survive.
            end = index + 1
            while end < length and text[end] in " \t":
                end += 1
            out.append("\\n")
            out.append(text[index + 1 : end])
            index = end
            continue
        if following.isdigit():
            out.append("\\n")
            index += 2
            continue
        out.append("\\\\")
        index += 1
    return "".join(out)


# Provider failures that say nothing about the conversation: sending the same
# request again is the right response. Everything else (a validation error, a
# bad request, a missing model) is surfaced.
_TRANSIENT_PROVIDER_ERRORS = frozenset(
    {
        "InternalServerError",
        "ServiceUnavailableError",
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "RateLimitError",
        "TryAgain",
        # The provider stopped mid-response; the sample is unusable and there
        # is nothing for the model to correct, so the request is resent.
        "IncompleteOutputException",
    }
)

_TRUNCATED_OUTPUT = "incomplete due to a max_tokens length limit"


def _returned_no_content(exc: BaseException) -> bool:
    """True when the provider answered with no content at all.

    Some upstreams spend the whole response on reasoning and emit nothing, or
    stop after a single token. There is nothing for the model to correct, so
    the sample is replaced rather than fed back.
    """
    from instructor.core.exceptions import InstructorRetryException

    if not isinstance(exc, InstructorRetryException):
        return False
    completion = _completion_object(exc)
    if completion is None:
        return False
    return _message_content(completion) is None


def _returned_no_envelope(exc: BaseException, result_fields: frozenset[str]) -> bool:
    """True when the provider answered with JSON that is not this protocol's.

    A model occasionally returns an object from some other API — a chat
    request, a wrapper around the conversation — instead of the response
    envelope. That is not a content mistake to correct: the model was not
    answering the question at all, so resend the request and let the gateway
    route to a healthy upstream. A response that carries the result's own
    fields is different; it is a correctable envelope mistake.
    """
    from instructor.core.exceptions import InstructorRetryException

    if not isinstance(exc, InstructorRetryException):
        return False
    text = _failed_attempt_text(exc)
    if text is None:
        return False
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    return "intent" not in data and not (set(data) & result_fields)


def _is_transient_provider_error(exc: BaseException) -> bool:
    """True when resending the same request could succeed.

    Walks the cause chain so a wrapped provider error is still recognised, and
    falls back to the HTTP status when the client does not expose a named type.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _TRANSIENT_PROVIDER_ERRORS:
            return True
        if _TRUNCATED_OUTPUT in str(current).lower():
            return True
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True
        current = current.__cause__ or current.__context__
    return False


def _recover_from_mangled_escape(
    exc: Exception, response_model: Any
) -> tuple[Any | None, ValidationFailure | None]:
    """Repair a mangled escape, then parse the response or report its errors.

    When the repaired text still does not validate, the field errors are
    returned so the model is told what is actually wrong instead of being
    sent back to fix an escape it cannot see.
    """
    content = _failed_attempt_text(exc)
    if content is None:
        return None, None
    repaired = _repair_json_escapes(content)
    if repaired == content:
        return None, None
    try:
        return response_model.model_validate_json(repaired), None
    except Exception as validation_error:
        return None, ValidationFailure(
            reason=str(validation_error).strip(),
            rejected_text=repaired,
        )



