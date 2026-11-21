#!/usr/bin/env python3
import unittest

from pydantic import TypeAdapter, ValidationError

from chat_workflow import AgentIntent, AgentResponse, Continue, Failure, Success, TurnResult
from workflows.evaluation_criteria import Criterion, EvaluationCriteria


class TestCriterionModel(unittest.TestCase):
    def test_criterion_creation(self):
        criterion = Criterion(
            name="budget",
            description="Budget constraint",
            weight=8.0,
            ideal_value="Under $100",
        )

        self.assertEqual(criterion.name, "budget")
        self.assertEqual(criterion.description, "Budget constraint")
        self.assertEqual(criterion.weight, 8.0)
        self.assertEqual(criterion.ideal_value, "Under $100")

    def test_criterion_default_weight(self):
        criterion = Criterion(name="quality", description="Quality level")

        self.assertEqual(criterion.name, "quality")
        self.assertEqual(criterion.weight, 1.0)
        self.assertIsNone(criterion.ideal_value)

    def test_criterion_weight_constraints(self):
        Criterion(name="test", description="test", weight=0.0)
        Criterion(name="test", description="test", weight=5.0)
        Criterion(name="test", description="test", weight=10.0)

        with self.assertRaises(ValidationError):
            Criterion(name="test", description="test", weight=-1.0)

        with self.assertRaises(ValidationError):
            Criterion(name="test", description="test", weight=11.0)


class TestEvaluationCriteriaModel(unittest.TestCase):
    def setUp(self):
        self.valid_criteria = [
            Criterion(name="budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]

    def test_evaluation_criteria_creation(self):
        criteria = EvaluationCriteria(context="Choosing a laptop", criteria=self.valid_criteria)

        self.assertEqual(criteria.context, "Choosing a laptop")
        self.assertEqual(len(criteria.criteria), 2)
        self.assertEqual(criteria.criteria[0].name, "budget")
        self.assertEqual(criteria.criteria[1].name, "quality")

    def test_evaluation_criteria_default_context(self):
        criteria = EvaluationCriteria(criteria=self.valid_criteria)

        self.assertEqual(criteria.context, "General decision making")
        self.assertEqual(len(criteria.criteria), 2)

    def test_business_rule_at_least_two_criteria(self):
        EvaluationCriteria(criteria=self.valid_criteria)

        more_criteria = self.valid_criteria + [Criterion(name="features", description="Feature set", weight=6.0)]
        EvaluationCriteria(criteria=more_criteria)

        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            EvaluationCriteria(criteria=[self.valid_criteria[0]])

    def test_business_rule_must_include_budget(self):
        EvaluationCriteria(criteria=self.valid_criteria)

        budget_uppercase = [
            Criterion(name="Budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        EvaluationCriteria(criteria=budget_uppercase)

        budget_allcaps = [
            Criterion(name="BUDGET", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        EvaluationCriteria(criteria=budget_allcaps)

        no_budget = [
            Criterion(name="cost", description="Cost constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        # A rule stated on the answer model is a validation failure of that
        # answer: Pydantic wraps the raised rule so the framework's output
        # retry can hand its sentence back to the model.
        with self.assertRaises(ValidationError) as context:
            EvaluationCriteria(criteria=no_budget)

        self.assertIn("Must include a criterion named 'budget'", str(context.exception))

    def test_add_criterion_method(self):
        criteria = EvaluationCriteria(criteria=self.valid_criteria)

        criteria.add_criterion(
            name="features",
            description="Feature set",
            weight=6.0,
            ideal_value="Many useful features",
        )

        self.assertEqual(len(criteria.criteria), 3)
        self.assertEqual(criteria.criteria[2].name, "features")
        self.assertEqual(criteria.criteria[2].description, "Feature set")
        self.assertEqual(criteria.criteria[2].weight, 6.0)
        self.assertEqual(criteria.criteria[2].ideal_value, "Many useful features")

    def test_total_weight_method(self):
        criteria = EvaluationCriteria(
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
                Criterion(name="features", description="Features", weight=6.0),
            ]
        )

        self.assertEqual(criteria.total_weight(), 21.0)

    def test_normalized_weights_method(self):
        criteria = EvaluationCriteria(
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=2.0),
            ]
        )

        normalized = criteria.normalized_weights()

        self.assertEqual(len(normalized), 2)
        self.assertAlmostEqual(normalized[0], 0.8)
        self.assertAlmostEqual(normalized[1], 0.2)
        self.assertAlmostEqual(sum(normalized), 1.0)

    def test_normalized_weights_zero_total(self):
        criteria = EvaluationCriteria(
            criteria=[
                Criterion(name="budget", description="Budget", weight=0.0),
                Criterion(name="quality", description="Quality", weight=0.0),
            ]
        )

        normalized = criteria.normalized_weights()

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[1], 0.0)


class TestAgentResponseShapes(unittest.TestCase):
    """The three shapes are the schema: each carries exactly its own fields."""

    def test_conversation_action_continue(self):
        action = Continue(intent=AgentIntent.CONTINUE, message="What's your budget?")

        self.assertEqual(action.intent, AgentIntent.CONTINUE)
        self.assertEqual(action.message, "What's your budget?")

    def test_conversation_action_success(self):
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

        action = Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=criteria,
            )

        self.assertEqual(action.intent, AgentIntent.SUCCESS)
        self.assertEqual(action.result, criteria)

    def test_conversation_action_failure(self):
        action = Failure(intent=AgentIntent.FAILURE, message="Can't help with that")

        self.assertEqual(action.intent, AgentIntent.FAILURE)
        self.assertEqual(action.message, "Can't help with that")

    def test_the_schema_forbids_mixing_the_shapes(self):
        """`continue` has no result; `success` carries words and the object.

        A model validator could only reject such an answer after it was
        written, and Pydantic never serialises validator logic into the
        schema the model reads. The union puts the rule in the shape: the
        forbidden field is absent, and `extra="forbid"` rejects it if sent.
        """
        branches = {
            "continue": (Continue, ("message",)),
            "success": (Success[EvaluationCriteria], ("message", "result")),
            "failure": (Failure, ("message",)),
        }
        for intent, (variant, expected) in branches.items():
            schema = TypeAdapter(variant).json_schema()
            self.assertEqual(set(schema["properties"]), {"intent", *expected}, intent)
            self.assertFalse(schema["additionalProperties"], intent)

        with self.assertRaises(ValueError):
            TypeAdapter(AgentResponse[EvaluationCriteria]).validate_python(
                {"intent": "continue", "message": "Which?", "result": {"context": "x", "criteria": []}}
            )

        # A success is a turn: it cannot finish without words.
        with self.assertRaises(ValueError):
            TypeAdapter(AgentResponse[EvaluationCriteria]).validate_python(
                {
                    "intent": "success",
                    "result": EvaluationCriteria(
                        context="test",
                        criteria=[
                            Criterion(
                                name="budget", description="Budget", weight=8.0
                            ),
                            Criterion(
                                name="quality", description="Quality", weight=7.0
                            ),
                        ],
                    ).model_dump(),
                }
            )

    def test_each_shape_requires_its_own_fields(self):
        with self.assertRaises(ValueError):
            Continue(intent=AgentIntent.CONTINUE, message="")
        with self.assertRaises(ValueError):
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=None,
            )
        with self.assertRaises(ValueError):
            Failure(intent=AgentIntent.FAILURE, message="")

class TestTurnResultModel(unittest.TestCase):
    def setUp(self):
        self.valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

    def test_continuing_factory_method(self):
        result = TurnResult[EvaluationCriteria].continuing("Please tell me more")

        self.assertEqual(result.message, "Please tell me more")
        self.assertIsNone(result.result)
        self.assertFalse(result.is_complete)

    def test_success_factory_method(self):
        result = TurnResult[EvaluationCriteria].success(self.valid_criteria)

        self.assertEqual(result.message, "Completed successfully!")
        self.assertEqual(result.result, self.valid_criteria)
        self.assertTrue(result.is_complete)

    def test_failure_factory_method(self):
        result = TurnResult[EvaluationCriteria].failure("Maximum turns reached")

        self.assertEqual(result.message, "Maximum turns reached")
        self.assertIsNone(result.result)
        self.assertTrue(result.is_complete)

    def test_direct_creation(self):
        result = TurnResult[EvaluationCriteria](result=self.valid_criteria, message="Custom message", is_complete=True)

        self.assertEqual(result.result, self.valid_criteria)
        self.assertEqual(result.message, "Custom message")
        self.assertTrue(result.is_complete)


if __name__ == "__main__":
    unittest.main(verbosity=2)
