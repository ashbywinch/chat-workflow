"""Pydantic models for conversation actions and results."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

TResult = TypeVar("TResult")


class ConversationAction(BaseModel, Generic[TResult]):
    action: Literal["continue", "success", "failure"]
    message: str | None = Field(
        default=None,
        description=(
            'Message for the user. Required when action is "continue" or "failure". '
            'Must be null when action is "success".'
        ),
    )
    result: TResult | None = Field(
        default=None,
        description=(
            'The criteria object. Required when action is "success". '
            'Must be null when action is "continue" or "failure".'
        ),
    )

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action == "continue":
            if not self.message:
                raise ValueError(
                    "continue action requires a message field with your question for the user. "
                    "Do not include a result field."
                )
            if self.result is not None:
                raise ValueError(
                    "continue action cannot include result. "
                    "Use action='success' if you have complete criteria to return."
                )
        elif self.action == "failure":
            if not self.message:
                raise ValueError("failure action requires a message field explaining why.")
            if self.result is not None:
                raise ValueError("failure action cannot include result.")
        elif self.action == "success":
            if self.result is None:
                raise ValueError("success action requires a result field with the complete criteria.")
        return self


class ConversationResult(BaseModel, Generic[TResult]):
    result: TResult | None = None
    message: str
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> ConversationResult[TResult]:
        return cls(result=None, message=message, is_complete=False)

    @classmethod
    def success(
        cls,
        result: TResult,
        message: str = "Completed successfully!",
    ) -> ConversationResult[TResult]:
        return cls(result=result, message=message, is_complete=True)

    @classmethod
    def failure(cls, message: str) -> ConversationResult[TResult]:
        return cls(result=None, message=message, is_complete=True)