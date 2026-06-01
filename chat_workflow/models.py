"""Pydantic models for agent responses and workflow turn results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

TResult = TypeVar("TResult")


class AgentIntent(StrEnum):
    """What the agent wants the framework to do next."""

    CONTINUE = "continue"
    SUCCESS = "success"
    FAILURE = "failure"


class AgentResponse(BaseModel, Generic[TResult]):
    """The structured response from an LLM agent after one turn.

    The agent tells the framework what to do next via ``intent``:
    - ``CONTINUE``: ask the user another question
    - ``SUCCESS``: return the completed result
    - ``FAILURE``: abort with an error message
    """

    intent: AgentIntent
    message: str | None = Field(
        default=None,
        description=(
            'Message for the user. Required when intent is "continue" or "failure". '
            'Must be null when intent is "success".'
        ),
    )
    result: TResult | None = Field(
        default=None,
        description=(
            'The final object. Required when intent is "success". Must be null when intent is "continue" or "failure".'
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _give_llm_actionable_feedback(cls, data: Any) -> Any:
        """Catch common LLM output mistakes and give actionable error messages.

        - Bare list: LLM returned an array without AgentResponse wrapper.
        - Inner fields directly: LLM returned domain fields without intent/result.
        """
        if isinstance(data, list):
            raise ValueError(
                "Received a bare list/array instead of an AgentResponse object. "
                "You must wrap your data with intent and result fields, for example:\n"
                '  AgentResponse(intent="success", result=<your_data>)'
            )
        if isinstance(data, dict):
            known_inner_fields = {
                "consumer",
                "format",
                "success_criteria",
                "integration_points",
                "storage_requirements",
                "source",
                "trigger_conditions",
                "dependencies",
                "validation_criteria",
                "phases",
                "activities",
            }
            if known_inner_fields & data.keys() and "intent" not in data and "result" not in data:
                raise ValueError(
                    "Received only inner/domain fields without an AgentResponse wrapper. "
                    "Your data includes fields like '{}' but is missing the required "
                    "'intent' and 'result' keys. Wrap your response, for example:\n"
                    '  AgentResponse(intent="success", result=<your_domain_data>)'.format(
                        "', '".join(sorted(known_inner_fields & data.keys()))
                    )
                )
        return data

    @model_validator(mode="after")
    def validate_intent_consistency(self):
        if self.intent == AgentIntent.CONTINUE:
            if not self.message:
                raise ValueError(
                    "CONTINUE intent requires a message field with your question for the user. "
                    "Do not include a result field."
                )
            # Tolerate result with CONTINUE — some LLMs (e.g. Gemini Flash Lite)
            # persistently include a partial result. The on_continue callback
            # only uses message, so the result is safely ignored.
        elif self.intent == AgentIntent.FAILURE:
            if not self.message:
                raise ValueError("FAILURE intent requires a message field explaining why.")
            if self.result is not None:
                raise ValueError("FAILURE intent cannot include result.")
        elif self.intent == AgentIntent.SUCCESS and self.result is None:
            raise ValueError("SUCCESS intent requires a result field with the complete object.")
        return self


class TurnResult(BaseModel, Generic[TResult]):
    """The outcome of processing a single turn in an atomic workflow."""

    result: TResult | None = None
    message: str
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> TurnResult[TResult]:
        """The workflow should continue with another turn."""
        return cls(result=None, message=message, is_complete=False)

    @classmethod
    def success(
        cls,
        result: TResult,
        message: str = "Completed successfully!",
    ) -> TurnResult[TResult]:
        """The workflow completed with a valid result."""
        return cls(result=result, message=message, is_complete=True)

    @classmethod
    def failure(cls, message: str) -> TurnResult[TResult]:
        """The workflow failed with an error message."""
        return cls(result=None, message=message, is_complete=True)
