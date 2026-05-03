from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class Criterion(BaseModel):
    """A single criterion for evaluating options."""

    name: str = Field(
        ..., description="Name of the criterion (e.g., 'Educational Value', 'Safety')"
    )
    description: str = Field(
        ..., description="Detailed description of what this criterion measures"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Importance weight from 0.0 (not important) to 10.0 (critical)",
    )
    ideal_value: Optional[str] = Field(
        default=None,
        description="Ideal or target value for this criterion (e.g., 'High quality materials', 'Under $50')",
    )


class EvaluationCriteria(BaseModel):
    """A list of criteria for evaluating possible choices."""

    criteria: List[Criterion] = Field(
        default_factory=list, description="List of evaluation criteria"
    )
    context: str = Field(
        default="General decision making",
        description="Context for these evaluation criteria (e.g., 'Birthday presents for a 7-year-old child')",
    )

    @model_validator(mode="after")
    def validate_business_rules(self):
        """Validate business rules for EvaluationCriteria."""
        from prompt_core.exceptions import CriteriaValidationError

        if len(self.criteria) < 2:
            raise CriteriaValidationError("Must have at least 2 criteria")

        if not any(c.name.lower() == "budget" for c in self.criteria):
            raise CriteriaValidationError(
                "Must include a criterion named 'budget' (case-insensitive)"
            )
        return self

    def add_criterion(
        self,
        name: str,
        description: str,
        weight: float = 1.0,
        ideal_value: Optional[str] = None,
    ):
        self.criteria.append(
            Criterion(
                name=name,
                description=description,
                weight=weight,
                ideal_value=ideal_value,
            )
        )

    def total_weight(self) -> float:
        return sum(criterion.weight for criterion in self.criteria)

    def normalized_weights(self) -> List[float]:
        total = self.total_weight()
        if total == 0:
            return [0.0] * len(self.criteria)
        return [criterion.weight / total for criterion in self.criteria]
