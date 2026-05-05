"""EvaluationCriteria workflow functions using the prompt-core framework."""

from __future__ import annotations

from typing import Annotated, TypeVar
from prompt_core import chat, workflow, ConversationTools
from pydantic import BaseModel

from .models import EvaluationCriteria
from .presentation import print_criteria


@chat
def generate_criteria(
    context: Annotated[
        str, "The topic or domain for which to generate evaluation criteria"
    ] = "",
    max_turns: Annotated[
        int, "Maximum number of conversation turns before giving up"
    ] = 10,
) -> EvaluationCriteria:
    """You are a helpful assistant guiding the user to create evaluation criteria.

    Behavior:
    - Ask one question at a time.
    - Start broad, then ask specific follow-ups.
    - Base output only on information explicitly provided by the user.
    - If the user is vague, ask clarifying questions.
    - If the user is uncooperative or refuses to provide useful information, use action="failure".

    Output actions:
    - action="continue": ask the next helpful question.
    - action="success": return complete criteria with action.result.
    - action="failure": end the conversation when useful criteria cannot be produced.
    """
    pass


ModelType = TypeVar("ModelType", bound=BaseModel)


@chat
def refine(
    initial_object: Annotated[
        ModelType, "The object to review and potentially modify based on user feedback"
    ],
    max_turns: Annotated[int, "Maximum number of refinement turns"] = 5,
) -> ModelType:
    """You are running a short refinement conversation for an existing object.
    Goal: Check whether the user wants to keep this version of the object or change anything about it. Return the object with any updates.

    Rules:
    - Ask one question at a time.
    - Use only user-provided feedback.
    - Preserve the original object contents exactly unless the user asks to change it.

    Response actions:
    - action="continue": further questions are required.
    - action="success": return the object (updated or unchanged).
    - action="failure": only if the user refuses to engage.

    Here is the current object to review:
    {initial_object.model_dump()}
    """
    pass


@workflow
def generate_reviewed_criteria(
    context: str = "", max_turns: int = 10, *, tools: ConversationTools
) -> EvaluationCriteria:
    criteria = generate_criteria(context=context, max_turns=max_turns, tools=tools)

    while True:
        print_criteria(
            criteria=criteria,
            title="Current criteria:",
            echo=tools.io.echo,
        )

        refined = refine(
            initial_object=criteria,
            max_turns=max_turns,
            tools=tools,
        )

        if refined.model_dump() == criteria.model_dump():
            return refined

        criteria = refined
