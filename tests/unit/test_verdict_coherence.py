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

    def test_a_negated_violation_is_a_pass(self):
        """'does not violate' is a pass conclusion, not a failure phrase."""
        verdict = RuleVerdict(
            rule="The consumer is a party, not the artifact",
            passed=False,
            explanation="The consumer names a role, so the data does not violate the rule.",
            evidence="consumer: the billing team",
        )

        self.assertTrue(_verdict_claims_the_rule_passes(verdict))

    def test_a_shouted_pass_conclusion_is_discarded(self):
        """A judge that argues its way to PASS and still says false is malformed."""
        verdict = RuleVerdict(
            rule="The assistant does not repeat a question",
            passed=False,
            explanation=(
                "The assistant asked a different follow-up, so this is not a "
                "repetition of the earlier question. So PASS."
            ),
            evidence="Is there anything else?",
        )

        self.assertTrue(_verdict_claims_the_rule_passes(verdict))
        self.assertFalse(_is_actionable_failure(verdict))

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

    def test_rule_says_not_violated_but_passed_is_false(self):
        """The judge wrote a pass conclusion and a 'no fix needed' correction."""
        verdict = RuleVerdict(
            rule="If the name alone is ambiguous, the description explains what it contains",
            passed=False,
            explanation=(
                "The name 'Meeting Minutes' is not ambiguous. Since the name is "
                "unambiguous, the rule is not violated."
            ),
            evidence="description: null",
            fix="No fix needed; the name is unambiguous.",
        )

        self.assertTrue(_verdict_claims_the_rule_passes(verdict))
        self.assertFalse(_is_actionable_failure(verdict))

    def test_a_narrated_correction_does_not_hide_the_pass(self):
        """The judge concluded a pass, then narrated withdrawing its verdict."""
        verdict = RuleVerdict(
            rule=(
                "The scope states what this part of the work itself does; it "
                "also states at least one thing it does not do."
            ),
            passed=False,
            explanation=(
                "However, it does not state at least one thing it does not do. "
                "Evidence: 'Does not include capturing the raw notes during "
                "the meeting.' Fix: The scope_description already includes that "
                "statement. Re-evaluating: the data does state it. So this "
                "actually passes. I need to correct my verdict."
            ),
            evidence="scope_description: 'Does not include capturing the raw notes.'",
            fix=(
                "The scope_description already includes the out-of-scope "
                "statement, so this actually passes. I need to correct my "
                "verdict."
            ),
        )

        self.assertTrue(_verdict_claims_the_rule_passes(verdict))
        self.assertFalse(_is_actionable_failure(verdict))

    def test_a_narration_alone_keeps_a_real_failure(self):
        """Skipping verdict narration must not read the previous sentence as a pass."""
        verdict = RuleVerdict(
            rule="The name is a noun naming a concrete artifact",
            passed=False,
            explanation=(
                "The name 'GenerateInvoice' is an action, so it fails the rule. "
                "I need to correct my verdict."
            ),
            evidence="name: GenerateInvoice",
            fix="replace name 'GenerateInvoice' with 'Invoice'",
        )

        self.assertFalse(_verdict_claims_the_rule_passes(verdict))
        self.assertTrue(_is_actionable_failure(verdict))

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
