"""Eval tests for Component._validation_rules — Tier 1 (Core Structural Integrity).

Each test constructs a Component with data that SHOULD violate a specific rule.
Before the rule is added to _validation_rules, construction succeeds (RED).
After the rule is added, validate_llm_rules catches the violation (GREEN).
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from chat_workflow.exceptions import ValidationError
from workflows.workflow.component import Component


class TestComponentSingleArtifactTypeRule(unittest.TestCase):
    """Rule: Component defines exactly one business artifact type."""

    def test_violation_multiple_artifact_types_rejected(self):
        """Purpose describing two artifact types should be rejected."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            with self.assertRaises(ValidationError):
                Component(
                    name="InvoiceManager",
                    purpose="Invoice processing pipeline with integrated timesheet management",
                    code_path=p,
                    model_class="InvoiceManager",
                    expert_role="Invoice Processing Specialist",
                    component_type="artifact_producing",
                )

    def test_happy_path_single_artifact_accepted(self):
        """Purpose describing one artifact type should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="InvoiceManager",
                purpose="Processes customer invoices through their complete lifecycle",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentSingleResponsibilityRule(unittest.TestCase):
    """Rule: Responsibility stateable in one sentence without conjunctions."""

    def test_violation_multiple_responsibilities_rejected(self):
        """Purpose describing multiple distinct responsibilities should be rejected."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            with self.assertRaises(ValidationError):
                Component(
                    name="InvoiceManager",
                    purpose="Oversees customer onboarding handles billing inquiries manages support tickets",
                    code_path=p,
                    model_class="InvoiceManager",
                    expert_role="Invoice Processing Specialist",
                    component_type="artifact_producing",
                )

    def test_happy_path_single_responsibility_accepted(self):
        """Purpose describing one clear responsibility should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="InvoiceManager",
                purpose="Processes customer invoices through their complete lifecycle",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentNoMultipleArtifactCreation(unittest.TestCase):
    """Rule: Component creates exactly one primary artifact type."""

    def test_violation_multiple_artifacts_rejected(self):
        """Purpose implying creation of multiple distinct artifacts should be rejected."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            with self.assertRaises(ValidationError):
                Component(
                    name="DataManager",
                    purpose="Creates customer invoices generates monthly reports produces analytics dashboards",
                    code_path=p,
                    model_class="DataManager",
                    expert_role="Data Management Specialist",
                    component_type="artifact_producing",
                )

    def test_happy_path_single_artifact_accepted(self):
        """Purpose describing creation of one artifact should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="InvoiceManager",
                purpose="Creates customer invoices from start to finish",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentClearBoundaries(unittest.TestCase):
    """Rule: Purpose clearly defines what's inside vs outside responsibility."""

    def test_violation_vague_boundaries_rejected(self):
        """Vague purpose without clear boundaries should be rejected."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            with self.assertRaises(ValidationError):
                Component(
                    name="SystemManager",
                    purpose="Handles everything related to the business operations of the company",
                    code_path=p,
                    model_class="SystemManager",
                    expert_role="System Administrator",
                    component_type="artifact_producing",
                )

    def test_happy_path_clear_boundaries_accepted(self):
        """Purpose with clear inside/outside boundaries should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="InvoiceManager",
                purpose="Creates customer invoices from submission through final distribution",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentEncapsulation(unittest.TestCase):
    """Rule: All fields/methods relate to the same domain concept."""

    def test_violation_unrelated_concerns_rejected(self):
        """Purpose mixing unrelated domain concerns should be rejected."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            with self.assertRaises(ValidationError):
                Component(
                    name="DataHub",
                    purpose="Manages user authentication database backups email notifications",
                    code_path=p,
                    model_class="DataHub",
                    expert_role="Data Hub Administrator",
                    component_type="artifact_producing",
                )

    def test_happy_path_focused_domain_accepted(self):
        """Purpose focused on one domain concept should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="InvoiceManager",
                purpose="Creates customer invoices from submission through final distribution",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentCohesion(unittest.TestCase):
    """Rule: All serve the same primary artifact."""

    def test_violation_orphaned_functionality_rejected(self):
        """Purpose with functionality not serving one primary artifact should be rejected."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            with self.assertRaises(ValidationError):
                Component(
                    name="TaskMaster",
                    purpose="Handles email notifications performs database maintenance",
                    code_path=p,
                    model_class="TaskMaster",
                    expert_role="Task Master Administrator",
                    component_type="artifact_producing",
                )

    def test_happy_path_cohesive_functionality_accepted(self):
        """Purpose where all functionality serves one artifact should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component(
                name="InvoiceManager",
                purpose="Creates customer invoices from submission through final distribution",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


if __name__ == "__main__":
    unittest.main(verbosity=2)
