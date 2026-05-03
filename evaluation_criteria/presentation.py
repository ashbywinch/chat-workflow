"""Presentation helpers for EvaluationCriteria conversations."""

from typing import Callable

from .models import EvaluationCriteria


def print_criteria(
    criteria: EvaluationCriteria,
    title: str,
    echo: Callable[[str], None],
) -> None:
    echo(f"\n{title}")
    echo(f"✓ Generated {len(criteria.criteria)} criteria")
    echo(f"Context: {criteria.context}")

    for i, criterion in enumerate(criteria.criteria, 1):
        echo(f"\n{i}. {criterion.name} (weight: {criterion.weight})")
        echo(f"   Description: {criterion.description}")
        if criterion.ideal_value:
            echo(f"   Ideal: {criterion.ideal_value}")

    echo("\nNormalized weights (sum to 1.0):")
    normalized = criteria.normalized_weights()
    for criterion, weight in zip(criteria.criteria, normalized):
        echo(f"  {criterion.name}: {weight:.3f}")
