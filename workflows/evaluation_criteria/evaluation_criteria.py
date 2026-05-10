from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator

from chat_workflow import chat


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


class EvaluationCriteria(BaseModel):
    """A list of criteria for evaluating possible choices in a specific context.

    For example, evaluating options for a planned purchase.

    Args:
        criteria: List of evaluation criteria
        context: Context for these evaluation criteria
            (e.g., 'Birthday presents for a 7-year-old child').
    """

    criteria: list[Criterion] = Field(
        default_factory=list,
        description="List of evaluation criteria (minimum 2 required, must include one named 'budget')",
        min_length=2,
    )
    context: str = Field(
        default="General decision making",
        description="Context for these evaluation criteria (e.g., 'Birthday presents for a 7-year-old child')",
    )

    @chat
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
        - Ask one question at a time.
        - Start broad, then ask specific follow-ups.
        - Base output only on information explicitly provided by the user.
        - If the user is vague, ask clarifying questions.
        - If the user is uncooperative or refuses to provide useful information within max_turns,
            use action="failure".
        """
        ...  # type: ignore[reportReturnType]

    @model_validator(mode="after")
    def validate_business_rules(self):
        """Validate business rules that can't be expressed as field attributes.

        Raises:
            ValidationError: If no criterion is named 'budget'.
        """
        from chat_workflow.exceptions import ValidationError

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
