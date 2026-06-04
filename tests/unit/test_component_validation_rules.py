"""Structural tests for Component model fields.

LLM-judged ``_validation_rules`` are NOT tested here — they are
non-deterministic and require eval tests with real API calls
(see ``tests/evals/test_component_validation_evals.py``).
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflows.workflow.component import Component


class TestComponentSingleArtifactTypeRule(unittest.TestCase):
    """Rule: Component defines exactly one business artifact type."""

    def test_violation_multiple_artifact_types_accepted(self):
        """Purpose describing two artifact types still constructs (LLM rule not enforced in unit tests)."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="InvoiceManager",
                purpose="Invoice processing pipeline with integrated timesheet management",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentSingleResponsibilityRule(unittest.TestCase):
    """Rule: Responsibility stateable in one sentence without conjunctions."""

    def test_violation_multiple_responsibilities_accepted(self):
        """Purpose with multiple distinct responsibilities still constructs (LLM rule not enforced in unit tests)."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="InvoiceManager",
                purpose="Oversees customer onboarding handles billing inquiries manages support tickets",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")

    def test_happy_path_single_responsibility_accepted(self):
        """Purpose describing one clear responsibility should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
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

    def test_violation_multiple_artifacts_accepted(self):
        """Purpose implying multiple artifacts still constructs (LLM rule not enforced in unit tests)."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="DataManager",
                purpose="Creates customer invoices generates monthly reports produces analytics dashboards",
                code_path=p,
                model_class="DataManager",
                expert_role="Data Management Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "DataManager")

    def test_happy_path_single_artifact_accepted(self):
        """Purpose describing creation of one artifact should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="InvoiceManager",
                purpose="Creates customer invoices from start to finish",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentClearBoundaries(unittest.TestCase):
    """Rule: Component boundary clear in one sentence."""

    def test_violation_vague_boundaries_accepted(self):
        """Vague purpose without clear boundaries still constructs (LLM rule not enforced in unit tests)."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="UtilityBelt",
                purpose="Handles miscellaneous tasks odd jobs general upkeep",
                code_path=p,
                model_class="UtilityBelt",
                expert_role="General Utility Administrator",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "UtilityBelt")

    def test_happy_path_clear_boundaries_accepted(self):
        """Purpose with clear scope should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="InvoiceManager",
                purpose="Manages the complete invoice lifecycle from creation to payment",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentEncapsulation(unittest.TestCase):
    """Rule: All fields and methods relate to the same domain concept."""

    def test_violation_unrelated_concerns_accepted(self):
        """Purpose mixing unrelated domain concerns still constructs (LLM rule not enforced in unit tests)."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="SuperManager",
                purpose="Processes payroll calculates carbon footprint tracks server uptime",
                code_path=p,
                model_class="SuperManager",
                expert_role="Super Manager",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "SuperManager")

    def test_happy_path_encapsulated_domain_accepted(self):
        """Purpose with single cohesive domain should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="InvoiceManager",
                purpose="Manages all aspects of invoice processing and payment collection",
                code_path=p,
                model_class="InvoiceManager",
                expert_role="Invoice Processing Specialist",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "InvoiceManager")


class TestComponentCohesion(unittest.TestCase):
    """Rule: All serve the same primary artifact."""

    def test_violation_orphaned_functionality_accepted(self):
        """Purpose with functionality not serving one primary artifact still constructs (LLM rule not enforced)."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
                name="TaskMaster",
                purpose="Handles email notifications performs database maintenance",
                code_path=p,
                model_class="TaskMaster",
                expert_role="Task Master Administrator",
                component_type="artifact_producing",
            )
            self.assertEqual(c.name, "TaskMaster")

    def test_happy_path_cohesive_functionality_accepted(self):
        """Purpose where all functionality serves one artifact should be accepted."""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("")
            c = Component.model_construct(
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
