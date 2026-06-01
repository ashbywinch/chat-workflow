"""Eval tests for ComponentInteractionContext.gather() — Phase 3 of component creation.

Tests that the interaction context agent can understand how the user wants
the assistant to behave during artifact creation, through conversation with
a user who knows their domain but not code.
"""

import unittest

from tests.conftest import timeout


class TestInteractionGatheringEval(unittest.TestCase):
    """Eval tests for ComponentInteractionContext.gather()."""

    @timeout(120)
    def test_gather_interaction_context(self):
        """ComponentInteractionContext.gather() should probe for interaction preferences."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow import (
            ComponentDomainField,
            ComponentDomainSpec,
            ComponentInteractionContext,
            ComponentStructure,
            StructField,
            StructValidator,
        )

        domain_spec = ComponentDomainSpec(
            name="MeetingMinutes",
            description=(
                "Structured meeting minutes that capture what happened, what was decided, and what needs to happen next"
            ),
            fields=[
                ComponentDomainField(
                    name="meeting_date",
                    domain_description="The date the meeting took place",
                    field_type_hint="date",
                ),
                ComponentDomainField(
                    name="attendees",
                    domain_description="List of people who attended the meeting",
                    field_type_hint="list of person names",
                ),
                ComponentDomainField(
                    name="key_decisions",
                    domain_description="Important decisions made during the meeting",
                    field_type_hint="list of decision descriptions",
                ),
                ComponentDomainField(
                    name="action_items",
                    domain_description="Tasks assigned during the meeting with owners and due dates",
                    field_type_hint="list of action items",
                ),
            ],
            what_good_looks_like=[
                "Someone who missed the meeting can catch up in two minutes",
                "Every decision has enough context to be understood without the discussion",
                "Every action item has a clear owner and a due date",
            ],
            expert_role="Meeting Coordinator",
        )

        structure = ComponentStructure(
            description=(
                "Structured meeting minutes that capture what happened, what was decided, and what needs to happen next"
            ),
            base_class="BaseModel",
            fields=[
                StructField(
                    name="meeting_date",
                    type_expr="str",
                    field_def_kwargs={"description": "The date the meeting took place"},
                ),
                StructField(
                    name="attendees",
                    type_expr="list[str]",
                    field_def_kwargs={
                        "description": "People who attended the meeting",
                        "min_length": "1",
                    },
                ),
                StructField(
                    name="key_decisions",
                    type_expr="list[Decision]",
                    field_def_kwargs={"description": "Important decisions made during the meeting"},
                ),
                StructField(
                    name="action_items",
                    type_expr="list[ActionItem]",
                    field_def_kwargs={"description": "Tasks assigned during the meeting"},
                ),
            ],
            model_validators=[
                StructValidator(
                    rule="Every decision must include enough context to stand on its own",
                    domain_origin=("Someone who missed the meeting can catch up in two minutes"),
                ),
                StructValidator(
                    rule="Every action item must have both an owner and a due date",
                    domain_origin=("Every action item has a clear owner and a due date"),
                ),
            ],
        )

        user_persona = (
            "You are a busy meeting coordinator who manages weekly team meetings. "
            "You have been doing this for years and have strong opinions about how "
            "the process should work.\n\n"
            "You want the assistant helping you create meeting minutes to always "
            "capture decisions with enough context — you hate it when someone reads "
            "the minutes later and can't figure out why a decision was made. You also "
            "want the assistant to proactively suggest action item owners based on "
            "what was discussed, because you often forget who volunteered for what. "
            "You prefer a professional but friendly tone — this is work, not a chat "
            "with friends. And you've noticed that people often forget to list all "
            "attendees or write action items without clear owners, so you want the "
            "assistant to watch for those mistakes.\n\n"
            "But you know absolutely NOTHING about programming, prompts, code, or "
            "technical concepts. If the person you're talking to uses technical jargon, "
            "ask them to explain in plain language.\n\n"
            "Respond helpfully but don't repeat yourself. If you already told someone "
            "something, don't re-confirm it. If they propose something reasonable, "
            "confirm it and move on."
        )

        judge_rules = {
            "Probes for priorities": (
                "The agent asked what the assistant should always prioritize or address "
                "when helping create the artifact, phrased in domain terms."
            ),
            "Probes for proactive suggestions": (
                "The agent asked what the assistant should suggest proactively — things "
                "the user might not think to ask about but would appreciate being reminded of."
            ),
            "Probes for pain points": (
                "The agent asked about common mistakes or frustrations the user has "
                "experienced when creating this artifact."
            ),
            "Uses domain language not code": (
                "The agent phrased all questions in domain terms — talking about what "
                "the assistant should do when helping create the artifact, never about "
                "prompts, code, or implementation."
            ),
            "Proposes then confirms": (
                "The agent proposed a complete picture of interaction preferences for "
                "the user to confirm or adjust, rather than asking about each topic "
                "one question at a time."
            ),
            "No repetition": (
                "The agent didn't ask for the same information again after the user "
                "already provided it. Asking for more detail after a vague answer is "
                "fine — repeating the exact same question after a complete answer is not."
            ),
        }

        result = run_multi_turn_eval(
            model_method=ComponentInteractionContext.gather,
            method_kwargs=dict(
                domain_spec=domain_spec,
                structure=structure,
                max_turns=10,
            ),
            user_persona=user_persona,
            judge_rules=judge_rules,
        )
        self.assertIsInstance(result, ComponentInteractionContext)
        # Should have at least some interaction preferences gathered
        self.assertGreaterEqual(len(result.must_prioritize), 1)
        self.assertGreaterEqual(len(result.auto_suggest), 1)
        self.assertGreaterEqual(len(result.user_pain_points), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
