"""Pydantic models for agent responses and workflow turn results."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AgentAnswer",
    "AgentIntent",
    "AgentResponse",
    "Continue",
    "Failure",
    "FinalAnswer",
    "FinalResponse",
    "Success",
    "TextReply",
    "TurnResult",
    "as_final",
    "as_turn",
]

TResult = TypeVar("TResult")


def _require_success_result(intent: AgentIntent, result: object, message: str) -> None:
    if intent is AgentIntent.SUCCESS and result is None:
        raise ValueError(message)


def _success_carries_its_result(self):
    _require_success_result(self.intent, self.result, self._RESULT_REQUIRED_MESSAGE)
    return self


class _CarriesResult:
    """A success answer must carry the object the step was asked for."""

    _RESULT_REQUIRED_MESSAGE: ClassVar[str] = (
        "A success must carry `result`: the complete object the step was asked for."
    )
    _a_success_carries_its_result = model_validator(mode="after")(_success_carries_its_result)


def _message_belongs_to_the_turn(self):
    if self.intent is AgentIntent.SUCCESS:
        if self.message is not None:
            raise ValueError(self._SUCCESS_HAS_NO_MESSAGE)
    elif not self.message:
        raise ValueError(self._TURN_NEEDS_A_MESSAGE)
    return self


class _MessageBelongs:
    """A success carries its result; a continue or failure carries words."""

    _SUCCESS_HAS_NO_MESSAGE: ClassVar[str] = (
        "A success carries its result and no message."
    )
    _TURN_NEEDS_A_MESSAGE: ClassVar[str] = (
        "A continue or failure must say what it is."
    )
    _a_message_belongs_to_the_turn = model_validator(mode="after")(
        _message_belongs_to_the_turn
    )


class AgentIntent(StrEnum):
    """What the agent wants the framework to do next."""

    CONTINUE = "continue"
    SUCCESS = "success"
    FAILURE = "failure"


class Continue(BaseModel):
    """Ask the user another question; the answer is not ready this turn."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.CONTINUE]
    message: str = Field(
        min_length=1,
        description=(
            "The question for the user, or a complete statement. Give any "
            "content you promise in this message; do not end on a lead-in to "
            "content that is not here."
        ),
    )


class Success(BaseModel, Generic[TResult]):
    """The finished object; the step is complete."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.SUCCESS]
    result: TResult = Field(description="The complete object the caller asked for.")


class Failure(BaseModel):
    """The result cannot be produced; stop and say why."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.FAILURE]
    message: str = Field(min_length=1, description="Why the result cannot be produced.")


type AgentResponse[TResult] = Annotated[
    Continue | Success[TResult] | Failure, Field(discriminator="intent")
]
"""The three turn shapes the loop switches on.

This union is *internal*. The model's tool schema is :class:`AgentAnswer` — one
flat object with ``intent``, an optional ``message`` and an optional
``result`` — because a union output type makes PydanticAI nest the answer under
``response`` while the prompt describes a top-level ``intent``. The model
follows the prose, the wrapper is missing, and retries are spent on a shape
nobody asked for. ``as_turn`` builds these three shapes from the flat answer.
"""

type FinalResponse[TResult] = Annotated[
    Success[TResult] | Failure, Field(discriminator="intent")
]
"""The turn shapes for the last turn: no ``continue``, so no question.

The wire shape is :class:`FinalAnswer`, for the same reason.
"""


class AgentAnswer(_CarriesResult, _MessageBelongs, BaseModel, Generic[TResult]):
    """The wire answer: one flat object, the shape the protocol describes.

    A union output type makes PydanticAI wrap the answer under ``response``,
    while the prompt speaks of a top-level ``intent``. The model follows the
    prose and the wrapper is missing, so every call burns retries on a shape
    nobody asked for. This model *is* the protocol: the tool schema and the
    prompt describe the same object, and the three turn shapes are made here,
    once, from it.
    """

    model_config = ConfigDict(extra="forbid")

    # The name of the tool that carries a step's answer. The endpoint enforces
    # a tool's schema where it ignores `response_format`, so the answer is a call.
    TOOL_NAME: ClassVar[str] = "answer"

    intent: AgentIntent = Field(
        description="continue while another turn is needed; success with the "
        "result; failure if it cannot be produced."
    )
    message: str | None = Field(
        default=None,
        description="The question on a continue, or the reason on a failure. "
        "A success carries its result and no message.",
    )
    result: TResult | None = Field(
        default=None,
        description="The complete object, on a success; absent otherwise.",
    )


class FinalAnswer(_CarriesResult, _MessageBelongs, BaseModel, Generic[TResult]):
    """The last turn's wire answer: the result, or the reason it is impossible."""

    _RESULT_REQUIRED_MESSAGE: ClassVar[str] = (
        "This is the final turn. A success must carry the complete result. "
        "There is no further turn to ask the user: apply the user's last "
        "correction, if any, and return the result you built from what they "
        "gave; return failure with the reason only if a required value cannot "
        "be produced."
    )
    _TURN_NEEDS_A_MESSAGE: ClassVar[str] = (
        "A failure must say why the result cannot be produced."
    )

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.SUCCESS, AgentIntent.FAILURE]
    message: str | None = Field(
        default=None,
        description="The reason on a failure. A success carries its result and "
        "no message.",
    )
    result: TResult | None = Field(default=None)


def _success_from_result(result: TResult | None) -> Success[TResult]:
    if result is None:
        raise ValueError("A success must carry `result`; the caller needs the object.")
    return Success(intent=AgentIntent.SUCCESS, result=result)


def _failure_from_message(message: str | None) -> Failure:
    if message is None:
        raise ValueError("A failure must carry its reason.")
    return Failure(intent=AgentIntent.FAILURE, message=message)


def as_turn(answer: AgentAnswer[TResult]) -> Continue | Success[TResult] | Failure:
    """Turn the flat wire answer into the three-shape turn the loop uses."""
    if answer.intent is AgentIntent.CONTINUE:
        if answer.message is None:
            raise ValueError("A continue must carry its message.")
        return Continue(intent=AgentIntent.CONTINUE, message=answer.message)
    if answer.intent is AgentIntent.SUCCESS:
        return _success_from_result(answer.result)
    return _failure_from_message(answer.message)


def as_final(answer: FinalAnswer[TResult]) -> Success[TResult] | Failure:
    """Turn the flat final answer into a success or failure."""
    if answer.intent is AgentIntent.SUCCESS:
        return _success_from_result(answer.result)
    return _failure_from_message(answer.message)


class TextReply(BaseModel):
    """One piece of user-facing text, such as a greeting or a closing."""

    text: str = Field(
        min_length=1,
        description="The text to show the user.",
    )


class TurnResult(BaseModel, Generic[TResult]):
    """The outcome of processing a single turn in an atomic workflow."""

    result: TResult | None = None
    message: str | None = None
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> TurnResult[TResult]:
        """The workflow should continue with another turn."""
        return cls(result=None, message=message, is_complete=False)

    @classmethod
    def success(cls, result: TResult) -> TurnResult[TResult]:
        """The workflow completed with a valid result; there is no message."""
        return cls(result=result, is_complete=True)

    @classmethod
    def failure(cls, message: str) -> TurnResult[TResult]:
        """The workflow failed with an error message."""
        return cls(result=None, message=message, is_complete=True)