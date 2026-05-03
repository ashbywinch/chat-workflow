"""EvaluationCriteria workflow functions using the prompt-core framework."""

from __future__ import annotations

import json

from prompt_core import leaf, workflow, ConversationTools

from .models import EvaluationCriteria
from .presentation import print_criteria


@leaf
def generate_criteria(context: str = "", max_turns: int = 10) -> EvaluationCriteria:
    """You are a helpful assistant guiding the user to create evaluation criteria.
    You have up to {max_turns} turns.

    Behavior:
    - Ask one question at a time.
    - Start broad, then ask specific follow-ups.
    - Base output only on information explicitly provided by the user.
    - If the user is vague, ask clarifying questions.
    - If the user is uncooperative or refuses to provide useful information, use action="failure".

    Output actions:
    - action="continue": ask the next helpful question.
    - action="success": return complete criteria with action.result.
    - action="failure": end the conversation when useful criteria cannot be produced (no result needed).

    When returning action="success", your criteria MUST include a criterion named "budget" (lowercase). This is a validation requirement - criteria without "budget" will be rejected.

    Context: {context}
    """
    pass


@leaf
def refine_criteria(
    initial_criteria: EvaluationCriteria, max_turns: int = 5
) -> EvaluationCriteria:
    """You are running a short refinement conversation for existing evaluation criteria.
    Goal: help the user keep or update the criteria and return a final result.

    Rules:
    - Ask one question at a time.
    - Use only user-provided feedback.
    - Preserve the original context unless the user asks to change it.
    - Turn limit: {max_turns} total turns.

    Response actions:
    - action="continue": ask one focused follow-up question.
    - action="success": return final criteria with action.result (updated or unchanged).
    - action="failure": only if the user refuses to engage.

    When returning action="success", your criteria MUST include a criterion named "budget" (lowercase). This is a validation requirement - criteria without "budget" will be rejected.

    Here is the current criteria to review:
    {criteria_json}
    """
    pass


@workflow
def run_reviewed_criteria_conversation(
    context: str = "", max_turns: int = 10, *, tools: ConversationTools
) -> EvaluationCriteria:
    initial = generate_criteria(context=context, max_turns=max_turns, tools=tools)

    print_criteria(
        criteria=initial,
        title="Initial criteria:",
        echo=tools.io.echo,
    )

    tools.io.echo(
        "\nAssistant: What do you think about these criteria? "
        "We can keep them or update them."
    )

    criteria_json = json.dumps(initial.model_dump(), indent=2)
    final = refine_criteria(
        initial_criteria=initial,
        max_turns=max_turns,
        criteria_json=criteria_json,
        tools=tools,
    )

    print_criteria(
        criteria=final,
        title="Final criteria:",
        echo=tools.io.echo,
    )

    return final
