"""EvaluationCriteria workflow functions using the chat-workflow framework."""

from __future__ import annotations

from typing import Annotated, TypeVar

from pydantic import BaseModel

from chat_workflow import atomic_workflow

ModelType = TypeVar("ModelType", bound=BaseModel)


@atomic_workflow
def refine(
    initial_object: Annotated[ModelType, "The object to review and potentially modify based on user feedback"],
) -> ModelType:
    """You are running a short refinement conversation for an existing object.
    Goal: Check whether the user wants to keep this version of the object or change
    anything about it. Return the object with any updates.

    Rules:
    - Ask one question at a time.
    - Use only user-provided feedback.
    - Preserve the original object contents exactly unless the user asks to change it.

    """
    # The @atomic_workflow decorator replaces this body with the
    # implementation; it is never executed.
    raise NotImplementedError("refine is implemented by @atomic_workflow")
