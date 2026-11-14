"""Unit tests for judge-verdict coherence.

A verdict that says the rule passes in its explanation but sets ``passed=False``
is malformed; it must not fail the object.
"""

import unittest

from chat_workflow.mixins import (
    RuleVerdict,
    _is_actionable_failure,
    _verdict_claims_the_rule_passes,
)


class TestVerdictCoherence(unittest.TestCase):
    def test_contradictory_failure_is_discarded(self):
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation=(
                "The name 'Raw Meeting Notes' is a noun phrase naming a "
                "concrete artifact, so it passes the rule."
            ),
            evidence="name: Raw Meeting Notes",
        )

        self.assertTrue(_verdict_claims_the_rule_passes(verdict))

    def test_real_failure_is_kept(self):
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation="The name is an action, not a noun; it fails the rule.",
            evidence="name: GenerateInvoice",
        )

        self.assertFalse(_verdict_claims_the_rule_passes(verdict))

    def test_negated_pass_phrase_is_not_discarded(self):
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation="It does not pass the rule because it is an action.",
            evidence="name: GenerateInvoice",
        )

        self.assertFalse(_verdict_claims_the_rule_passes(verdict))

    def test_technically_satisfied_conclusion_is_discarded(self):
        verdict = RuleVerdict(
            rule="If the name alone is ambiguous, the description explains it",
            passed=False,
            explanation=(
                "The name is not ambiguous, so the rule is technically "
                "satisfied. The name is not ambiguous, so the rule passes."
            ),
            evidence='name: "Meeting minutes", description: null',
        )

        self.assertTrue(_verdict_claims_the_rule_passes(verdict))
        self.assertFalse(_is_actionable_failure(verdict))

    def test_failure_without_fix_is_not_actionable(self):
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation="The name is an action, not a noun; it fails the rule.",
            evidence="name: GenerateInvoice",
            fix=None,
        )

        self.assertFalse(_is_actionable_failure(verdict))

    def test_failure_without_evidence_is_not_actionable(self):
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation="The name is an action, not a noun; it fails the rule.",
            evidence=None,
            fix="replace name 'GenerateInvoice' with 'Invoice'",
        )

        self.assertFalse(_is_actionable_failure(verdict))

    def test_substantive_failure_is_actionable(self):
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation="The name is an action, not a noun; it fails the rule.",
            evidence="name: GenerateInvoice",
            fix="replace name 'GenerateInvoice' with 'Invoice'",
        )

        self.assertTrue(_is_actionable_failure(verdict))


if __name__ == "__main__":
    unittest.main(verbosity=2)
