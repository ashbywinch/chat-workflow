import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from .annotations import UserLabel, user_labels
from .atomic_workflow_step_config import AtomicWorkflowStepConfig
from .conversation_rules import FRAMEWORK_TERMS
from .debug import _DebugTimer
from .exceptions import AtomicWorkflowStepFailedError
from .llm_interaction import ProviderType
from .models import AgentIntent, AgentResponse, TurnResult
from .session_log import TranscriptEvent

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
    kind: str = "validation_correction"

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
        self._request_timeout_seconds = config.request_timeout_seconds
        self._model_supports_tools = config.model_supports_tools
        self._api_base: str | None = config.api_base
        self._api_key_env: str | None = config.api_key_env
        self.chat = config.chat
        self.response_model = config.response_model
        # The decorator always supplies these callbacks (they are only optional
        # on the config so a hand-built config can omit them before setup).
        self.on_continue: Callable[[AgentResponse[TResult]], TurnResult[TResult]] = cast(
            Callable[[AgentResponse[TResult]], TurnResult[TResult]], config.on_continue
        )
        self.on_success: Callable[[AgentResponse[TResult]], TurnResult[TResult]] = cast(
            Callable[[AgentResponse[TResult]], TurnResult[TResult]], config.on_success
        )
        self.on_failure: Callable[[AgentResponse[TResult]], AtomicWorkflowStepFailedError] = cast(
            Callable[[AgentResponse[TResult]], AtomicWorkflowStepFailedError], config.on_failure
        )
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
                TranscriptEvent(type="user", content=user_input, turn=self.turn_count + 1)
            )

        self.turn_count += 1
        response = self._call_llm()

        message = response.message
        if message and response.intent != AgentIntent.SUCCESS:
            self.messages.append({"role": "assistant", "content": message})
            self.events.append(
                TranscriptEvent(type="assistant", content=message, turn=self.turn_count)
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
            error.messages = list(self.messages)
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
                response = self._request(client)
            except InstructorRetryException as exc:
                failure = _validation_failure(exc)
                if failure is None or attempt >= self._validation_corrections:
                    raise MaxRetriesExceededError(
                        self._max_retries,
                        failure.message() if failure else str(exc),
                    ) from exc
                self._apply_correction(failure, attempt + 1)
                continue
            except Exception as exc:
                if self.debug:
                    self.debug.on_error(exc)
                raise

            failure = self._language_failure(response) or self._vocabulary_failure(
                response
            )
            if failure is not None:
                if attempt >= self._validation_corrections:
                    raise MaxRetriesExceededError(self._max_retries, failure.message())
                self._apply_correction(failure, attempt + 1)
                continue

            self._mark_recovered()
            return response

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
            # The model sometimes drifts into Chinese on long prompts. A final
            # system instruction is more reliable than the task-section line.
            messages.append(
                {
                    "role": "system",
                    "content": "The conversation is in English. Reply in English.",
                }
            )
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
            )
        timer.emit_response(response)
        return response

    def _language_failure(
        self, response: AgentResponse[TResult]
    ) -> ValidationFailure | None:
        message = response.message or ""
        if not _is_wrong_language(_language_reference(self.messages), message):
            return None
        return ValidationFailure(
            reason=(
                "Reply in the user's language. The user is writing in English; "
                "write your reply in English and do not switch languages on "
                "your own."
            ),
            rejected_text=message,
            kind="language_correction",
        )

    def _vocabulary_failure(
        self, response: AgentResponse[TResult]
    ) -> ValidationFailure | None:
        """Reject a message that leaks an internal label the user never used."""
        labels = user_labels(_inner_result_model(self.response_model))
        if not labels and not FRAMEWORK_TERMS:
            return None
        message = response.message or ""
        user_text = _user_text(self.messages)
        candidates: dict[str, UserLabel] = {}
        for name, label in labels.items():
            candidates[name] = label
            for alias in label.aliases:
                candidates[alias] = label
        for term, phrase in FRAMEWORK_TERMS.items():
            candidates.setdefault(term, UserLabel(phrase, term))
        for term, label in candidates.items():
            if _mentions(message, term) and not _mentions(user_text, term):
                return ValidationFailure(
                    reason=(
                        f"Your message used the internal label '{term}'. "
                        "The user has not used that word."
                    ),
                    guidance=(
                        f'Rewrite the message without "{term}". Use the '
                        f'user\'s own words, or say "{label.phrase}". Keep '
                        "the result unchanged."
                    ),
                    rejected_text=message,
                    kind="vocabulary_correction",
                )
        return None

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
                    meta={"kind": "rejected_response"},
                )
            )
        correction = _correction_prompt(failure)
        # A correction is a framework instruction, not a user turn. Sending it
        # as a system message keeps the model from answering the previous
        # question and ignoring the format requirement.
        self.messages.append({"role": "system", "content": correction})
        self.events.append(
            TranscriptEvent(
                type="framework",
                content=correction,
                turn=self.turn_count,
                meta={"kind": failure.kind},
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


def _is_wrong_language(reference: str, response: str) -> bool:
    """True when an English conversation gets a Chinese reply.

    Only CJK drift is detected. A handful of characters is allowed (a name or
    a quoted term); five or more means the model switched languages. A user
    who writes in Chinese makes the reference Chinese, so the guard stays out
    of the way.
    """
    cjk = sum(1 for char in response if "\u4e00" <= char <= "\u9fff")
    if cjk < 5:
        return False
    reference_cjk = sum(1 for char in reference if "\u4e00" <= char <= "\u9fff")
    return reference_cjk == 0


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
    reason = str(cause).strip()
    guidance = next(
        (
            text
            for needles, text in _FAILURE_GUIDANCE
            if any(needle in reason for needle in needles)
        ),
        None,
    )
    return ValidationFailure(
        reason=reason,
        guidance=guidance,
        rejected_text=_completion_text(exc),
    )


def _correction_prompt(failure: ValidationFailure) -> str:
    """The framework turn that tells the model how to repair its response."""
    if failure.kind in ("language_correction", "vocabulary_correction"):
        return f"[framework] {failure.message()}"
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


def _completion_text(exc: Exception) -> str | None:
    """Return the assistant content from the last rejected completion."""
    completion = getattr(exc, "last_completion", None)
    if completion is None:
        attempts = getattr(exc, "failed_attempts", None) or []
        if attempts:
            completion = getattr(attempts[-1], "completion", None)
    choices = getattr(completion, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return None



