#!/usr/bin/env python3
import unittest
from pydantic import ValidationError

from evaluation_criteria.models import EvaluationCriteria, Criterion
from prompt_core import ConversationAction, ConversationResult


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
        criteria = EvaluationCriteria(
            context="Choosing a laptop", criteria=self.valid_criteria
        )

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

        more_criteria = self.valid_criteria + [
            Criterion(name="features", description="Feature set", weight=6.0)
        ]
        EvaluationCriteria(criteria=more_criteria)

        from prompt_core.exceptions import CriteriaValidationError

        with self.assertRaises(CriteriaValidationError) as context:
            EvaluationCriteria(criteria=[self.valid_criteria[0]])

        self.assertIn("Must have at least 2 criteria", str(context.exception))

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

        from prompt_core.exceptions import CriteriaValidationError

        no_budget = [
            Criterion(name="cost", description="Cost constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        with self.assertRaises(CriteriaValidationError) as context:
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


class TestConversationActionModel(unittest.TestCase):
    def test_conversation_action_continue(self):
        action = ConversationAction[EvaluationCriteria](
            action="continue", message="What's your budget?"
        )

        self.assertEqual(action.action, "continue")
        self.assertEqual(action.message, "What's your budget?")
        self.assertIsNone(action.result)

    def test_conversation_action_success(self):
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

        action = ConversationAction[EvaluationCriteria](
            action="success", result=criteria
        )

        self.assertEqual(action.action, "success")
        self.assertIsNone(action.message)
        self.assertEqual(action.result, criteria)

    def test_conversation_action_failure(self):
        action = ConversationAction[EvaluationCriteria](
            action="failure", message="Can't help with that"
        )

        self.assertEqual(action.action, "failure")
        self.assertEqual(action.message, "Can't help with that")
        self.assertIsNone(action.result)

    def test_action_validation_continue_without_message(self):
        with self.assertRaises(ValueError) as context:
            ConversationAction[EvaluationCriteria](action="continue", message=None)

        self.assertIn("continue action requires a message", str(context.exception))

    def test_action_validation_success_without_result(self):
        with self.assertRaises(ValueError) as context:
            ConversationAction[EvaluationCriteria](action="success", result=None)

        self.assertIn("success action requires a result", str(context.exception))

    def test_action_validation_failure_without_message(self):
        with self.assertRaises(ValueError) as context:
            ConversationAction[EvaluationCriteria](action="failure", message=None)

        self.assertIn("failure action requires a message", str(context.exception))


class TestConversationResultModel(unittest.TestCase):
    def setUp(self):
        self.valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

    def test_continuing_factory_method(self):
        result = ConversationResult[EvaluationCriteria].continuing(
            "Please tell me more"
        )

        self.assertEqual(result.message, "Please tell me more")
        self.assertIsNone(result.result)
        self.assertFalse(result.is_complete)

    def test_success_factory_method(self):
        result = ConversationResult[EvaluationCriteria].success(self.valid_criteria)

        self.assertEqual(result.message, "Completed successfully!")
        self.assertEqual(result.result, self.valid_criteria)
        self.assertTrue(result.is_complete)

    def test_failure_factory_method(self):
        result = ConversationResult[EvaluationCriteria].failure("Maximum turns reached")

        self.assertEqual(result.message, "Maximum turns reached")
        self.assertIsNone(result.result)
        self.assertTrue(result.is_complete)

    def test_direct_creation(self):
        result = ConversationResult[EvaluationCriteria](
            result=self.valid_criteria, message="Custom message", is_complete=True
        )

        self.assertEqual(result.result, self.valid_criteria)
        self.assertEqual(result.message, "Custom message")
        self.assertTrue(result.is_complete)


if __name__ == "__main__":
    unittest.main(verbosity=2)
