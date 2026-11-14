import copy
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from .atomic_workflow_step_config import AtomicWorkflowStepConfig
from .debug import _DebugTimer
from .llm_interaction import ProviderType
from .models import AgentIntent, AgentResponse, TurnResult
from .session_log import TranscriptEvent

TResult = TypeVar("TResult")


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

        from .exceptions import MaxRetriesExceededError, ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            client = get_client(
                provider=self._provider,
                api_key_env=self._api_key_env,
                api_base=self._api_base,
                model_supports_tools=self._model_supports_tools,
            )
        except ImportError as e:
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\nInstall litellm for multi-provider LLM support: uv add litellm"
            ) from e
        except Exception as e:
            if self.debug:
                self.debug.on_error(e)
            raise

        for correction in range(self._validation_corrections + 1):
            try:
                timer = _DebugTimer(self.debug, self.messages, self.model)

                # With corrections enabled we want one sample per call: the
                # framework feeds validation failures back in context. Letting
                # Instructor retry blind would repeat the same messages — and a
                # gateway may answer the identical prompt from cache, so the
                # model never sees the error. One sample per call makes every
                # correction an informed retry.
                instructor_retries = (
                    1 if self._validation_corrections > 0 else self._max_retries
                )

                with timer:
                    # Pass a copy of messages — Instructor patches messages
                    # in-place with the JSON schema. Using a copy keeps our
                    # conversation history clean across turns.
                    response = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                        model=self.model,
                        messages=copy.deepcopy(self.messages),  # pyright: ignore[reportArgumentType]
                        response_model=self.response_model,
                        max_retries=instructor_retries,
                        timeout=self._request_timeout_seconds,
                    )

                timer.emit_response(response)
                reference = _language_reference(self.messages)
                if _is_wrong_language(reference, response.message or ""):
                    reason = (
                        "Reply in the user's language. The user is writing in "
                        "English; write your reply in English and do not switch "
                        "languages on your own."
                    )
                    if correction >= self._validation_corrections:
                        raise MaxRetriesExceededError(self._max_retries, reason)
                    self.validation_events.append({
                        "turn": self.turn_count,
                        "attempt": correction + 1,
                        "reason": reason,
                        "recovered": False,
                    })
                    self.messages.append(
                        {"role": "assistant", "content": response.message or ""}
                    )
                    self.messages.append(
                        {"role": "user", "content": f"[framework] {reason}"}
                    )
                    self.events.append(
                        TranscriptEvent(
                            type="framework",
                            content=reason,
                            turn=self.turn_count,
                            meta={"kind": "language_correction"},
                        )
                    )
                    continue
                for event in self.validation_events:
                    if event["turn"] == self.turn_count:
                        event["recovered"] = True
                return response
            except InstructorRetryException as e:
                # A plain-text reply is a normal assistant turn that forgot the
                # JSON envelope. Accept it as `continue` rather than failing the
                # workflow; the next turn can still produce the structured
                # result. Malformed JSON is a real validation failure and is
                # retried below.
                plain_text = _plain_text_completion(e)
                if plain_text is not None:
                    return AgentResponse(
                        intent=AgentIntent.CONTINUE, message=plain_text
                    )
                reason = _retry_reason(e)
                if (
                    not _is_validation_failure(e)
                    or correction >= self._validation_corrections
                ):
                    raise MaxRetriesExceededError(self._max_retries, reason) from e
                self.validation_events.append({
                    "turn": self.turn_count,
                    "attempt": correction + 1,
                    "reason": reason,
                    "recovered": False,
                })
                # The failed assistant turn was never appended to the
                # conversation. Put it back so the model can see the code or
                # value it must correct, then append the rejection reason.
                rejected = _completion_text(e)
                if rejected:
                    self.messages.append({"role": "assistant", "content": rejected})
                    self.events.append(
                        TranscriptEvent(
                            type="framework",
                            content=rejected,
                            turn=self.turn_count,
                            meta={"kind": "rejected_response"},
                        )
                    )
                correction = (
                    "[framework] Your previous response was rejected by "
                    f"validation. Fix exactly these violations:\n\n{reason}\n\n"
                    "Keep everything that was valid; change only what the "
                    "violations name. Return the complete structured response "
                    "again as JSON only: "
                    '{"intent": "continue", "message": "<your reply to the '
                    'user>", "result": null} or '
                    '{"intent": "success", "message": null, "result": {...}}. '
                    "Put the user-facing text in `message`, never as plain "
                    "text. Do not reply with a description of the violations."
                )
                self.messages.append({"role": "user", "content": correction})
                self.events.append(
                    TranscriptEvent(
                        type="framework",
                        content=correction,
                        turn=self.turn_count,
                        meta={"kind": "validation_correction"},
                    )
                )
            except Exception as e:
                if self.debug:
                    self.debug.on_error(e)
                raise

        raise AssertionError("unreachable")


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


def _retry_reason(exc: Exception) -> str:
    """Extract the last failure message from an Instructor retry exception."""
    attempts = getattr(exc, "failed_attempts", None) or []
    if attempts:
        cause = getattr(attempts[-1], "exception", None)
        if cause is not None:
            reason = str(cause).strip()
            if "invalid escape" in reason:
                reason += (
                    " The JSON string has an invalid escape. Use `\\n` for every "
                    "newline (never a backslash followed by a space), escape a "
                    "literal backslash as `\\\\`, and do not wrap code in Markdown."
                )
            elif "invalid decimal literal" in reason or "expected an indented block" in reason:
                reason += (
                    " Write plain ASCII Python: remove Markdown fences, full-width "
                    "characters, and non-ASCII punctuation."
                )
            elif (
                "control character" in reason
                or "EOF while parsing" in reason
                or "Invalid JSON" in reason
            ):
                reason += (
                    " Return valid JSON only. Encode every newline inside a string "
                    "as \\n, never a literal line break; escape every double quote "
                    "inside a string as \\\"; close every brace and bracket."
                )
            return reason
    return str(exc).strip()


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


def _plain_text_completion(exc: Exception) -> str | None:
    """Return the completion when it is a plain-text assistant turn.

    The model occasionally answers in prose instead of the JSON envelope. That
    is a valid message to the user, not a corrupt result: treat it as
    ``continue``. Text that starts with ``{`` or ``[`` is a malformed JSON
    attempt and must still go through validation retries.
    """
    text = _completion_text(exc)
    if text is None:
        return None
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return None
    return text


def _is_validation_failure(exc: Exception) -> bool:
    """True when the last attempt failed validation or produced bad JSON."""
    from json import JSONDecodeError

    from pydantic import ValidationError as PydanticValidationError

    from .exceptions import ValidationError

    attempts = getattr(exc, "failed_attempts", None) or []
    if not attempts:
        return False
    cause = getattr(attempts[-1], "exception", None)
    return isinstance(cause, (PydanticValidationError, ValidationError, JSONDecodeError))
