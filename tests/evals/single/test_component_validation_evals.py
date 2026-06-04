"""Eval tests that exercise real LLM validation on the framework's own
classes via the Pydantic validator chain.

Individual rule-vs-violation tests use ``llm_judge()``; the integration
tests at the bottom read each framework class's own source file and
validate it through ``GeneratedComponent`` — proving the framework's
classes satisfy the rules they impose on generated code.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.conftest import timeout
from tests.evals.helpers import (
    JudgeResult,
    llm_judge,
    make_config,
)
from workflows.workflow.component_source_code import ComponentSourceCode

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class TestComponentValidationRulesEval(unittest.TestCase):
    """LLM-judge evals for each Component._validation_rule."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transcript(self, name: str, purpose: str, expert_role: str) -> str:
        """Format Component fields as a transcript for llm_judge."""
        return (
            f"Component Name: {name}\n"
            f"Purpose: {purpose}\n"
            f"Expert Role: {expert_role}"
        )

    # ==================================================================
    # Rule 1: Purpose describes domain concept (not just storage)
    #   _validation_rules[0]
    # ==================================================================

    _RULE_PURPOSE_DOMAIN: dict[str, str] = {
        "Purpose describes domain concept": (
            "The purpose field clearly describes the domain concept the component "
            "represents, in a way that an LLM could use to understand what instances "
            "of this component mean (not just what they store)."
        ),
    }

    @timeout(60)
    def test_purpose_describes_domain_concept_violation(self):
        """Purpose that merely describes data storage should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="InvoiceStore",
            purpose="Stores invoice data in a database table with fields for amount, date, and status",
            expert_role="Invoice Processing Clerk",
        )
        result: JudgeResult = llm_judge(self._RULE_PURPOSE_DOMAIN, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 2: Name is a noun
    #   _validation_rules[1]
    # ==================================================================

    _RULE_NAME_NOUN: dict[str, str] = {
        "Name is a noun": (
            "The name is a noun \u2014 it names an artifact (e.g. 'Invoice', "
            "'MeetingMinutes'), not an action (e.g. 'GenerateInvoice')."
        ),
    }

    @timeout(60)
    def test_name_is_noun_violation(self):
        """Name that is a verb phrase should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="GenerateInvoice",
            purpose="Creates invoices from start to finish",
            expert_role="Invoice Processing Clerk",
        )
        result: JudgeResult = llm_judge(self._RULE_NAME_NOUN, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 3: Expert role is specific (not generic)
    #   _validation_rules[2]
    # ==================================================================

    _RULE_EXPERT_SPECIFIC: dict[str, str] = {
        "Expert role is specific": (
            "The expert_role describes a real, specific domain expertise the component "
            "embodies, not a generic role ('MinutesDraft Expert' is vague; "
            "'Meeting Minutes Administrator' is better)."
        ),
    }

    @timeout(60)
    def test_expert_role_specific_violation(self):
        """Vague/generic expert role should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="Invoice",
            purpose="Creates invoices from start to finish",
            expert_role="Invoice Expert",
        )
        result: JudgeResult = llm_judge(self._RULE_EXPERT_SPECIFIC, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 4: Single Artifact Type Rule
    #   _validation_rules[3]
    # ==================================================================

    _RULE_SINGLE_ARTIFACT: dict[str, str] = {
        "Single Artifact Type Rule": (
            "The component defines exactly one business artifact type. "
            "The purpose must describe a single artifact concept. "
            "BAD: 'Invoice processing pipeline with integrated timesheet management' "
            "(two artifact types: invoices and timesheets). "
            "GOOD: 'Processes customer invoices through their complete lifecycle'."
        ),
    }

    @timeout(60)
    def test_single_artifact_type_violation(self):
        """Purpose describing two artifact types should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="InvoiceManager",
            purpose="Invoice processing pipeline with integrated timesheet management",
            expert_role="Invoice Processing Specialist",
        )
        result: JudgeResult = llm_judge(self._RULE_SINGLE_ARTIFACT, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 5: Single Responsibility
    #   _validation_rules[4]
    # ==================================================================

    _RULE_SINGLE_RESP: dict[str, str] = {
        "Single Responsibility": (
            "The component's responsibility must be stateable in one sentence "
            "describing exactly one domain concept. The purpose must not describe "
            "multiple distinct responsibilities. "
            "BAD: 'Oversees customer onboarding handles billing inquiries manages "
            "support tickets' (three distinct responsibilities). "
            "GOOD: 'Processes customer invoices through their complete lifecycle'."
        ),
    }

    @timeout(60)
    def test_single_responsibility_violation(self):
        """Purpose describing multiple responsibilities should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="InvoiceManager",
            purpose="Oversees customer onboarding handles billing inquiries manages support tickets",
            expert_role="Invoice Processing Specialist",
        )
        result: JudgeResult = llm_judge(self._RULE_SINGLE_RESP, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 6: No Multiple Artifact Creation
    #   _validation_rules[5]
    # ==================================================================

    _RULE_NO_MULTI_CREATE: dict[str, str] = {
        "No Multiple Artifact Creation": (
            "The component creates exactly one primary artifact type. "
            "The purpose must not imply creation of multiple distinct business artifacts. "
            "BAD: 'Creates customer invoices generates monthly reports produces "
            "analytics dashboards' (three distinct artifacts). "
            "GOOD: 'Creates customer invoices from start to finish'."
        ),
    }

    @timeout(60)
    def test_no_multiple_artifact_creation_violation(self):
        """Purpose implying creation of multiple distinct artifacts should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="DataManager",
            purpose="Creates customer invoices generates monthly reports produces analytics dashboards",
            expert_role="Data Management Specialist",
        )
        result: JudgeResult = llm_judge(self._RULE_NO_MULTI_CREATE, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 7: Clear Boundaries
    #   _validation_rules[6]
    # ==================================================================

    _RULE_CLEAR_BOUNDARIES: dict[str, str] = {
        "Clear Boundaries": (
            "The purpose must clearly define what is inside and outside the "
            "component's responsibility. Another LLM should be able to use the "
            "purpose to decide if a given concern belongs here. "
            "BAD: 'Handles everything related to the business operations of the "
            "company' (no clear boundary). "
            "GOOD: 'Creates customer invoices from submission through final "
            "distribution' (clear start and end)."
        ),
    }

    @timeout(60)
    def test_clear_boundaries_violation(self):
        """Vague purpose without clear boundaries should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="SystemManager",
            purpose="Handles everything related to the business operations of the company",
            expert_role="System Administrator",
        )
        result: JudgeResult = llm_judge(self._RULE_CLEAR_BOUNDARIES, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 8: Encapsulation
    #   _validation_rules[7]
    # ==================================================================

    _RULE_ENCAPSULATION: dict[str, str] = {
        "Encapsulation": (
            "The component's fields and methods must all relate to the same "
            "domain concept. The purpose must not mix unrelated concerns. "
            "BAD: 'Manages user authentication database backups email notifications' "
            "(three unrelated domains). "
            "GOOD: 'Creates customer invoices from submission through final "
            "distribution' (single domain)."
        ),
    }

    @timeout(60)
    def test_encapsulation_violation(self):
        """Purpose mixing unrelated domain concerns should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="DataHub",
            purpose="Manages user authentication database backups email notifications",
            expert_role="Data Hub Administrator",
        )
        result: JudgeResult = llm_judge(self._RULE_ENCAPSULATION, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Rule 9: Cohesion
    #   _validation_rules[8]
    # ==================================================================

    _RULE_COHESION: dict[str, str] = {
        "Cohesion": (
            "All functionality described in the purpose must serve the same "
            "primary artifact. The purpose must not describe orphaned functionality "
            "that belongs to a different domain. "
            "BAD: 'Handles email notifications performs database maintenance' "
            "(functionality serves no common artifact). "
            "GOOD: 'Creates customer invoices from submission through final "
            "distribution' (all serves invoice artifact)."
        ),
    }

    @timeout(60)
    def test_cohesion_violation(self):
        """Purpose with orphaned functionality should FAIL."""
        config = make_config()
        transcript = self._transcript(
            name="TaskMaster",
            purpose="Handles email notifications performs database maintenance",
            expert_role="Task Master Administrator",
        )
        result: JudgeResult = llm_judge(self._RULE_COHESION, transcript, config)
        for v in result.verdicts:
            self.assertFalse(
                v.passed, f"Rule '{v.rule}' should have failed: {v.explanation}"
            )

    # ==================================================================
    # Self-validation: each framework class's source validated as code
    # ==================================================================

    @timeout(120)
    def test_component_source_passes_as_generated_code(self):
        """Component's own source code should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "component.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class Component", gc.code)

    @timeout(120)
    def test_workflow_source_passes_as_generated_code(self):
        """Workflow's own source code should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "workflow.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class Workflow", gc.code)

    @timeout(120)
    def test_generated_component_source_passes_as_generated_code(self):
        """ComponentSourceCode's own source should satisfy its own rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "component_source_code.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class ComponentSourceCode", gc.code)

    @timeout(120)
    def test_resource_source_passes_as_generated_code(self):
        """Resource model source should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "models" / "resource.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class Resource", gc.code)

    @timeout(120)
    def test_deliverable_source_passes_as_generated_code(self):
        """Deliverable model source should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "models" / "deliverable.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class Deliverable", gc.code)

    @timeout(120)
    def test_gap_analysis_source_passes_as_generated_code(self):
        """GapAnalysis model source should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "models" / "gap_analysis.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class GapAnalysis", gc.code)

    @timeout(120)
    def test_process_definition_source_passes_as_generated_code(self):
        """ProcessDefinition model source should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "models" / "process_definition.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class ProcessDefinition", gc.code)

    @timeout(120)
    def test_component_responsibilities_source_passes_as_generated_code(self):
        """ComponentResponsibilities source should satisfy GeneratedComponent rules."""
        class _GC(ComponentSourceCode):
            _skip_llm_validation = False

        code = (_PROJECT_ROOT / "workflows" / "workflow" / "component_responsibilities.py").read_text()
        gc = _GC(code=code)
        self.assertIn("class ComponentResponsibilities", gc.code)

if __name__ == "__main__":
    unittest.main(verbosity=2)
