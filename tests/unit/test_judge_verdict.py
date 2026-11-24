"""The judge's verdict is typed: pass or fail, with evidence.

Nothing about a verdict is read from prose. A judge that writes a correction,
an explanation field, or no evidence fails validation and is asked again.
"""

import unittest

from pydantic import ValidationError

from chat_workflow.mixins import (
    RuleFailed,
    RulePassed,
    RuleVerdicts,
    _is_actionable_failure,
)


class TestTypedVerdict(unittest.TestCase):
    def test_a_pass_is_not_a_failure(self):
        verdict = RulePassed(
            rule="The name is a noun", verdict="pass", evidence="name: Invoice"
        )

        self.assertFalse(_is_actionable_failure(verdict))

    def test_a_fail_with_evidence_is_actionable(self):
        verdict = RuleFailed(
            rule="The name is a noun", verdict="fail", evidence="name: GenerateInvoice"
        )

        self.assertTrue(_is_actionable_failure(verdict))

    def test_a_fail_without_evidence_is_rejected(self):
        with self.assertRaises(ValidationError):
            RuleFailed.model_validate(
                {"rule": "The name is a noun", "verdict": "fail"}
            )

    def test_a_pass_cannot_carry_a_correction(self):
        """A pass has nothing to fix; the judge does not propose repairs."""
        with self.assertRaises(ValidationError):
            RulePassed.model_validate(
                {
                    "rule": "The name is a noun",
                    "verdict": "pass",
                    "evidence": "name: Invoice",
                    "fix": "rename it",
                }
            )

    def test_a_verdict_must_say_pass_or_fail(self):
        with self.assertRaises(ValidationError):
            RuleVerdicts.model_validate(
                {"verdicts": [{"rule": "The name is a noun", "evidence": "x"}]}
            )

    def test_a_mixed_list_parses_by_its_verdicts(self):
        verdicts = RuleVerdicts.model_validate(
            {
                "verdicts": [
                    {"rule": "a", "verdict": "pass", "evidence": "a: 1"},
                    {"rule": "b", "verdict": "fail", "evidence": "b: null"},
                ]
            }
        )

        self.assertIsInstance(verdicts.verdicts[0], RulePassed)
        self.assertIsInstance(verdicts.verdicts[1], RuleFailed)

    def test_an_explanation_field_is_not_accepted(self):
        """The judge evaluates; the evidence is its explanation."""
        with self.assertRaises(ValidationError):
            RuleFailed.model_validate(
                {
                    "rule": "The name is a noun",
                    "verdict": "fail",
                    "evidence": "name: GenerateInvoice",
                    "explanation": "It is an action.",
                }
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
