"""EvaluationCriteria workflow functions using the chat-workflow framework."""

from __future__ import annotations

from chat_workflow import Session, composite_workflow

from .evaluation_criteria import EvaluationCriteria
from .formatter import echo_criteria
from .refine import refine


@composite_workflow
def generate_reviewed_criteria(
    context: str = "",
    max_turns: int = 10,
    max_refinements: int = 3,
    *,
    session: Session,
) -> EvaluationCriteria:
    criteria = EvaluationCriteria.generate_from_chat(context=context, max_turns=max_turns, session=session)

    for _ in range(max_refinements):
        echo_criteria(
            criteria,
            title="Current criteria:",
            echo=session.io.echo,
        )

        refined = refine(
            initial_object=criteria,
            max_turns=max_turns,
            session=session,
        )

        if refined.model_dump() == criteria.model_dump():
            return refined

        criteria = refined

    return criteria
