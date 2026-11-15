from __future__ import annotations

from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, Field, model_validator

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class Criterion(BaseModel):
    """A single criterion for evaluating options.

    Args:
        name: Name of the criterion (e.g., 'Educational Value', 'Safety').
        description: Detailed description of what this criterion measures.
        weight: Importance weight from 0.0 (not important) to 10.0 (critical).
        ideal_value: Ideal or target value for this criterion
            (e.g., 'High quality materials', 'Under $50').
    """

    name: str = Field(..., description="Name of the criterion (e.g., 'Educational Value', 'Safety')")
    description: str = Field(..., description="Detailed description of what this criterion measures")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Importance weight from 0.0 (not important) to 10.0 (critical)",
    )
    ideal_value: str | None = Field(
        default=None,
        description="Ideal or target value for this criterion (e.g., 'High quality materials', 'Under $50')",
    )
    priority_basis: str | None = Field(
        default=None,
        description=(
            "Why this weight: the user's own words or the use case that set "
            "this criterion's importance (e.g. 'works all day on a plane'). "
            "Not a restatement of the criterion."
        ),
    )


class EvaluationCriteria(LLMValidated):
    """A list of criteria for evaluating possible choices in a specific context.

    For example, evaluating options for a planned purchase.

    Args:
        criteria: List of evaluation criteria
        context: Context for these evaluation criteria
            (e.g., 'Birthday presents for a 7-year-old child').
    """

    _validation_rules: ClassVar[list[str]] = [
        "Each criterion is specific to the user's context, not a generic "
        "criterion that could apply to any decision.",
        "The weights reflect the user's stated priorities, not arbitrary "
        "defaults: every criterion carries a `priority_basis` grounded in what "
        "the user said, and the heaviest weight is the criterion they said "
        "matters most.",
        "The criteria cover the trade-offs the user actually faces.",
    ]

    criteria: list[Criterion] = Field(
        default_factory=list,
        description="List of evaluation criteria (minimum 2 required, must include one named 'budget')",
        min_length=2,
    )
    context: str = Field(
        default="General decision making",
        description="Context for these evaluation criteria (e.g., 'Birthday presents for a 7-year-old child')",
    )

    @atomic_workflow_step
    @classmethod
    def generate_from_chat(
        cls,
        context: Annotated[str, "The topic or domain for which to generate evaluation criteria"],
        max_turns: Annotated[int, "Maximum number of conversation turns before giving up"] = 10,
    ) -> Self:
        """You are a helpful assistant guiding the user to create evaluation criteria.
        Assume the user is an expert on their topic but they know nothing about
        creating good evaluation criteria.

        Behavior:
        - Use the context to frame the criteria.
        - Ask one question at a time.
        - Start broad, then ask specific follow-ups.
        - Base the criteria on what the user tells you. The one thing you
          infer is their relative importance.
        - If the user is vague, ask clarifying questions.
        - The criteria must include one named "budget". If the user has not
          mentioned cost, ask about it once in plain language; do not invent a
          number.
        - Infer the ranking from the use case they describe and state it in
          one sentence for confirmation. Never ask the user to rank the
          criteria one by one. Never give every criterion the same default
          weight unless the user says they matter equally.
        - Record why each weight is what it is in `priority_basis`, using the
          user's words or the use case they described.
        - When you have enough, or the user asks you to finalize, propose the
          complete set of criteria and return success in that same turn.
        - Use intent="failure" only if the user refuses to take part at all. A
          missing ranking, weight, or number is never a reason to fail: infer
          it and say what you assumed.
        """
        ...  # type: ignore[reportReturnType]

    @model_validator(mode="after")
    def validate_business_rules(self):
        """Validate business rules that can't be expressed as field attributes.

        Raises:
            ValidationError: If no criterion is named 'budget'.
        """
        from chat_workflow import ValidationError

        if not any(c.name.lower() == "budget" for c in self.criteria):
            raise ValidationError("Must include a criterion named 'budget' (case-insensitive)")
        return self

    def add_criterion(
        self,
        name: str,
        description: str,
        weight: float = 1.0,
        ideal_value: str | None = None,
    ):
        """Add a criterion to the evaluation criteria list.

        Args:
            name: Name of the criterion.
            description: Detailed description of what this criterion measures.
            weight: Importance weight from 0.0 to 10.0.
            ideal_value: Ideal or target value for this criterion.
        """
        self.criteria.append(
            Criterion(
                name=name,
                description=description,
                weight=weight,
                ideal_value=ideal_value,
            )
        )

    def total_weight(self) -> float:
        """Calculate the sum of all criterion weights.

        Returns:
            The sum of all criterion weight values.
        """
        return sum(criterion.weight for criterion in self.criteria)

    def normalized_weights(self) -> list[float]:
        """Calculate weight values normalized to sum to 1.0.

        Returns:
            A list of floats where each value is the criterion's weight
            divided by the total weight. Returns all zeros if total weight
            is zero.
        """
        total = self.total_weight()
        if total == 0:
            return [0.0] * len(self.criteria)
        return [criterion.weight / total for criterion in self.criteria]
