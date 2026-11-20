"""Pydantic models for agent responses and workflow turn results."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

TResult = TypeVar("TResult")

# The name of the tool that carries a step's answer. The endpoint enforces a
# tool's schema where it ignores `response_format`, so the answer is a call.
ANSWER_TOOL = "answer"


class AgentIntent(StrEnum):
    """What the agent wants the framework to do next."""

    CONTINUE = "continue"
    SUCCESS = "success"
    FAILURE = "failure"


class Continue(BaseModel):
    """Ask the user another question; the answer is not ready this turn."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.CONTINUE]
    message: str = Field(min_length=1, description="The question for the user.")


class Success(BaseModel, Generic[TResult]):
    """The finished object; the step is complete."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.SUCCESS]
    message: str = Field(
        min_length=1,
        description=(
            "What the user reads when the step ends: one or two sentences in "
            "their own words saying what you decided, what changed, or that it "
            "is confirmed. Do not list the result's values here."
        ),
    )
    result: TResult = Field(description="The complete object the caller asked for.")


class Failure(BaseModel):
    """The result cannot be produced; stop and say why."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[AgentIntent.FAILURE]
    message: str = Field(min_length=1, description="Why the result cannot be produced.")


type AgentResponse[TResult] = Annotated[
    Continue | Success[TResult] | Failure, Field(discriminator="intent")
]
"""What an agent returns after one turn, as three shapes, not one.

The schema *is* the rule: a ``continue`` has no ``result`` property, and a
``success`` carries both the words the user reads and the object, with
``additionalProperties: false`` on each branch. An answer that both asks and
answers is inexpressible, so it is never generated and never retried. A model
validator could not do this: Pydantic does not translate validator logic into
JSON Schema, so a rule written there reaches the model only as a rejection
after the fact.
"""

type FinalResponse[TResult] = Annotated[
    Success[TResult] | Failure, Field(discriminator="intent")
]
"""The answer when no turn remains: it cannot ask a question.

A step that has run out of turns is offered these shapes only, so "ask
another question" is not expressible and no retry has to argue about it.
"""


class TextReply(BaseModel):
    """One piece of user-facing text, such as a greeting or a closing."""

    text: str = Field(
        min_length=1,
        description="The text to show the user.",
    )


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