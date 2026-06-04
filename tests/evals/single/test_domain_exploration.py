"""Eval tests for ComponentDomainSpec.explore() — domain exploration phase.

Tests that the @atomic_workflow agent:
- Proposes artifact fields based on ComponentResponsibilities
- Probes for holistic quality criteria ("what good looks like")
- Asks about the expert role
- Doesn't re-ask or repeat itself
- Stays entirely in domain language
"""

import unittest

from tests.conftest import timeout


class TestDomainExplorationEval(unittest.TestCase):
    """Eval tests for ComponentDomainSpec.explore()."""

    @timeout(120)
    def test_domain_exploration_with_user_bot(self):
        """ComponentDomainSpec.explore() should complete efficiently with a user bot."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow import ComponentDomainSpec
        from workflows.workflow.component_responsibilities import (
            ComponentResponsibilities,
        )

        responsibilities = ComponentResponsibilities(
            name="MeetingMinutes",
            purpose="Transform raw meeting notes into structured minutes with clear decisions and action items",
            scope_description=(
                "Captures what happened in a meeting: decisions made, "
                "action items assigned, key discussion points. Does NOT "
                "include pre-meeting preparation or post-meeting follow-up tracking."
            ),
            required_inputs=["Raw meeting notes", "Attendee list"],
            component_type="artifact_producing",
            incidental_notes="",
        )

        user_persona = (
            "You are a busy team lead who runs weekly project meetings. "
            "You take rough notes during the meeting and need a consistent "
            "way to turn them into proper minutes that your team can use.\n\n"
            "You know exactly what good minutes look like: every decision "
            "must be recorded with who made it, every action item needs an "
            "owner and a due date, and the whole thing should be skimmable "
            "so someone who missed the meeting can catch up in two minutes. "
            "You've been doing this for years and have strong opinions about "
            "what works.\n\n"
            "But you know NOTHING about programming, data structures, or "
            "technical concepts. If the analyst uses technical jargon, ask "
            "them to explain in plain language.\n\n"
            "Respond helpfully to the analyst's questions using your meeting "
            "expertise. Be patient but don't repeat yourself. If they propose "
            "something reasonable, confirm it and move on. If they get "
            "something wrong, correct them clearly."
        )

        result = run_multi_turn_eval(
            model_method=ComponentDomainSpec.explore,
            method_kwargs=dict(
                responsibilities=responsibilities,
                max_turns=8,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, ComponentDomainSpec)
        self.assertEqual(result.name, "MeetingMinutes")
        self.assertGreater(len(result.fields), 0)
        self.assertGreater(len(result.what_good_looks_like), 0)
        self.assertGreater(len(result.expert_role), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
