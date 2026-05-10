"""EvaluationCriteria workflow functions using the chat-workflow framework."""

from __future__ import annotations

from chat_workflow import workflow
from chat_workflow.conversation_tools import ConversationTools

from .evaluation_criteria import EvaluationCriteria
from .formatter import echo_criteria
from .refine import refine


@workflow
def generate_reviewed_criteria(
    context: str = "",
    max_turns: int = 10,
    max_refinements: int = 3,
    *,
    tools: ConversationTools,
) -> EvaluationCriteria:
    criteria = EvaluationCriteria.generate_from_chat(context=context, max_turns=max_turns, tools=tools)

    for _ in range(max_refinements):
        echo_criteria(
            criteria,
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

    return criteria
