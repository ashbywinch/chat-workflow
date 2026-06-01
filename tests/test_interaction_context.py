"""Tests for ComponentInteractionContext model."""
import unittest

from pydantic import ValidationError

from workflows.workflow.interaction_context import ComponentInteractionContext


class TestComponentInteractionContext(unittest.TestCase):
    """Construction and field validation tests."""

    def make_valid(self, **overrides) -> ComponentInteractionContext:
        kwargs = dict(
            must_prioritize=[
                "Always ask about decisions early in the conversation"
            ],
            auto_suggest=[
                "Suggest action item owners based on the topic discussed"
            ],
            tone_preference="Professional but friendly",
            user_pain_points=["Users often forget to list attendees"],
        )
        kwargs.update(overrides)
        return ComponentInteractionContext(**kwargs)

    # --- Construction ---

    def test_valid_construction(self):
        ctx = self.make_valid()
        self.assertEqual(
            ctx.must_prioritize,
            ["Always ask about decisions early in the conversation"],
        )
        self.assertEqual(
            ctx.auto_suggest,
            ["Suggest action item owners based on the topic discussed"],
        )
        self.assertEqual(ctx.tone_preference, "Professional but friendly")
        self.assertEqual(
            ctx.user_pain_points, ["Users often forget to list attendees"]
        )

    def test_multiple_priorities(self):
        ctx = self.make_valid(
            must_prioritize=[
                "Ask about timeline first",
                "Clarify stakeholders",
                "Confirm budget constraints",
            ]
        )
        self.assertEqual(len(ctx.must_prioritize), 3)

    def test_multiple_suggestions(self):
        ctx = self.make_valid(
            auto_suggest=[
                "Suggest a follow-up meeting",
                "Recommend documentation tools",
            ]
        )
        self.assertEqual(len(ctx.auto_suggest), 2)

    def test_multiple_pain_points(self):
        ctx = self.make_valid(
            user_pain_points=[
                "Users forget attendees",
                "Users skip action items",
                "Users omit deadlines",
            ]
        )
        self.assertEqual(len(ctx.user_pain_points), 3)

    # --- tone_preference ---

    def test_tone_preference_can_be_none(self):
        ctx = self.make_valid(tone_preference=None)
        self.assertIsNone(ctx.tone_preference)

    def test_tone_preference_custom_value(self):
        ctx = self.make_valid(tone_preference="Casual and approachable")
        self.assertEqual(ctx.tone_preference, "Casual and approachable")

    # --- Empty lists ---

    def test_must_prioritize_can_be_empty(self):
        ctx = self.make_valid(must_prioritize=[])
        self.assertEqual(ctx.must_prioritize, [])

    def test_auto_suggest_can_be_empty(self):
        ctx = self.make_valid(auto_suggest=[])
        self.assertEqual(ctx.auto_suggest, [])

    def test_user_pain_points_can_be_empty(self):
        ctx = self.make_valid(user_pain_points=[])
        self.assertEqual(ctx.user_pain_points, [])

    # --- Missing required fields ---

    def test_missing_must_prioritize_raises(self):
        with self.assertRaises(ValidationError):
            ComponentInteractionContext(
                auto_suggest=["Suggest owners"],
                tone_preference="Professional",
                user_pain_points=["Forget attendees"],
            )

    def test_missing_auto_suggest_raises(self):
        with self.assertRaises(ValidationError):
            ComponentInteractionContext(
                must_prioritize=["Ask about decisions"],
                tone_preference="Professional",
                user_pain_points=["Forget attendees"],
            )

    def test_missing_user_pain_points_raises(self):
        with self.assertRaises(ValidationError):
            ComponentInteractionContext(
                must_prioritize=["Ask about decisions"],
                auto_suggest=["Suggest owners"],
                tone_preference="Professional",
            )

    # --- Edge cases ---

    def test_all_fields_empty_lists(self):
        ctx = self.make_valid(
            must_prioritize=[],
            auto_suggest=[],
            user_pain_points=[],
            tone_preference=None,
        )
        self.assertEqual(ctx.must_prioritize, [])
        self.assertEqual(ctx.auto_suggest, [])
        self.assertEqual(ctx.user_pain_points, [])
        self.assertIsNone(ctx.tone_preference)

    def test_long_list_values(self):
        long_item = "A very long description " * 50
        ctx = self.make_valid(must_prioritize=[long_item.strip()])
        self.assertTrue(ctx.must_prioritize[0].startswith("A very long"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
