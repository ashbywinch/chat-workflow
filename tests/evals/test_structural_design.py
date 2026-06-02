"""Eval tests for ComponentStructure.design() — Phase 2 of component creation.

Tests that the structural design agent can translate domain concepts into
Pydantic field definitions and validation rules through conversation with
a user who knows business rules but not code.
"""

import unittest

from tests.conftest import timeout


class TestStructuralDesignEval(unittest.TestCase):
    """Eval tests for ComponentStructure.design()."""

    @timeout(120)
    def test_design_from_domain_spec(self):
        """ComponentStructure.design() should propose types and validators from domain spec."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow import (
            ComponentDomainField,
            ComponentDomainSpec,
            ComponentStructure,
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
                ComponentDomainField(
                    name="next_meeting_date",
                    domain_description="When the next meeting is scheduled, if any",
                    field_type_hint="optional date",
                ),
            ],
            what_good_looks_like=[
                "Someone who missed the meeting can catch up in two minutes",
                "Every decision has enough context to be understood without the discussion",
                "Every action item has a clear owner and a due date",
                "The minutes are concise — no rambling or unnecessary detail",
            ],
            expert_role="Meeting Coordinator",
        )

        user_persona = (
            "You are a busy meeting coordinator who manages weekly team meetings. "
            "You know exactly what makes good meeting minutes: every decision needs "
            "enough context to stand on its own, every action item must have an owner "
            "and a due date, and someone who missed the meeting should be able to "
            "catch up in under two minutes. You also know that minutes should be "
            "concise — no rambling or unnecessary detail.\n\n"
            "But you know absolutely NOTHING about Python, Pydantic, data types, "
            "programming, or code. If the person you're talking to uses technical "
            "terms like 'string', 'list', 'field', 'validator', or 'type', ask them "
            "to explain in plain language.\n\n"
            "You are happy to confirm or adjust proposals about what information "
            "should be captured and what rules it should follow. For example, if "
            "someone asks 'should the description have a maximum length?' you can "
            "say 'yes, keep it to a sentence or two.' But if someone asks 'should "
            "I use max_length=200?' you should ask them to explain what that means.\n\n"
            "Respond helpfully but don't repeat yourself. If you already agreed to "
            "something, don't re-confirm it."
        )

        judge_rules = {
            "Proposes types from domain hints": (
                "The agent proposed concrete data types (like text, date, list of items) "
                "based on the domain field type hints, rather than asking the user what "
                "type each field should be."
            ),
            "Derives validators from quality criteria": (
                "The agent translated the 'what good looks like' criteria into specific "
                "validation rules or constraints. For example, 'every action item has a "
                "clear owner and a due date' became a rule about action items requiring "
                "both fields, or 'concise minutes' became a length constraint."
            ),
            "Uses domain language not code": (
                "The agent phrased all proposals and questions in business terms — "
                "'should the description have a maximum length?' not 'should I add "
                "max_length=200?' — and never used Python or Pydantic terminology "
                "with the user."
            ),
            "Proposes then confirms": (
                "The agent proposed a complete picture (types and rules) for the user "
                "to confirm or adjust, rather than asking about each field one at a time."
            ),
            "No repetition": (
                "The agent didn't ask for the same information again after the user "
                "already provided it. Asking for more detail after a vague answer is "
                "fine — repeating the exact same question after a complete answer is not."
            ),
        }

        result = run_multi_turn_eval(
            model_method=ComponentStructure.design,
            method_kwargs=dict(
                domain_spec=domain_spec,
                max_turns=5,
            ),
            user_persona=user_persona,
            judge_rules=judge_rules,
        )
        self.assertIsInstance(result, ComponentStructure)
        # Should have at least some fields and validators proposed
        self.assertGreaterEqual(len(result.fields), 1)
        self.assertGreaterEqual(len(result.model_validators), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
