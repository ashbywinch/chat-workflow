"""EvaluationCriteria-specific one-shot generation utilities."""

from typing import Optional

from prompt_core.llm_interaction import get_client

from .models import EvaluationCriteria


def generate_evaluation_criteria(
    context: str = "birthday presents for a child",
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> EvaluationCriteria:
    from prompt_core.config import config

    client = get_client()

    model = config.model
    temperature = temperature or config.temperature
    max_retries = max_retries or config.max_retries

    prompt = f"""
    You are an expert at creating evaluation criteria for decision making.
    
    Create a comprehensive set of criteria for evaluating options in this context: {context}
    
    For each criterion, provide:
    1. A clear name
    2. A detailed description of what it measures
    3. A weight from 0.0 to 10.0 indicating importance
    4. An ideal or target value if applicable
    
    The criteria should be specific, measurable, and relevant to the context.
    Include both objective and subjective criteria where appropriate.
    """

    criteria = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        timeout=config.request_timeout_seconds,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates structured evaluation criteria.",
            },
            {"role": "user", "content": prompt},
        ],
        response_model=EvaluationCriteria,
    )

    criteria.context = context
    return criteria
