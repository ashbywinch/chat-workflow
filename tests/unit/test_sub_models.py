"""Tests for workflow sub-models."""

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from workflows.workflow import GeneratedComponent
from workflows.workflow.models import (
    ComponentRequirement,
    GapAnalysis,
    Input,
    Output,
    ProcessAnalysis,
)


class TestProcessAnalysis(unittest.TestCase):
    def test_valid_instance(self):
        model = ProcessAnalysis(
            phases=["Intake", "Review"],
            activities=["Receive request", "Check completeness"],
            orchestrating_component="Case Management",
            participants=["Case Worker", "Manager"],
        )
        self.assertEqual(len(model.phases), 2)
        self.assertEqual(model.orchestrating_component, "Case Management")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ProcessAnalysis()  # no fields provided


class TestInput(unittest.TestCase):
    def test_valid_instance(self):
        model = Input(
            source="Customer",
            format="JSON",
            trigger_conditions="New request submitted",
            dependencies=["Customer profile"],
            validation_criteria="Must include customer ID",
        )
        self.assertEqual(model.source, "Customer")
        self.assertEqual(model.dependencies, ["Customer profile"])

    def test_default_dependencies(self):
        model = Input(
            source="System",
            format="CSV",
            trigger_conditions="Daily batch",
            validation_criteria="File must exist",
        )
        self.assertEqual(model.dependencies, [])

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Input()


class TestOutput(unittest.TestCase):
    def test_valid_instance(self):
        model = Output(
            consumer="Billing System",
            format="XML",
            success_criteria="All invoices processed",
            integration_points="API endpoint",
            storage_requirements="Database archive",
        )
        self.assertEqual(model.consumer, "Billing System")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Output()


class TestGapAnalysis(unittest.TestCase):
    def test_valid_instance(self):
        model = GapAnalysis(
            missing_components=["Payment Processor"],
            missing_playbooks=["Create Payment Playbook"],
            integration_gaps=["Billing \u2192 Payment handoff unclear"],
            organizational_gaps=["No owner for reconciliation"],
            recommendations=["Create Payment Processor component"],
        )
        self.assertEqual(len(model.missing_components), 1)

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            GapAnalysis()


class TestComponentRequirement(unittest.TestCase):
    def test_valid_instance(self):
        model = ComponentRequirement(
            name="Invoice",
            purpose="Manage invoice lifecycle",
            required_inputs=["Order details"],
            expected_outputs=["Invoice PDF"],
            component_type="artifact_producing",
        )
        self.assertEqual(model.name, "Invoice")
        self.assertEqual(model.component_type, "artifact_producing")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ComponentRequirement()


class TestGeneratedComponent(unittest.TestCase):
    @patch("chat_workflow.mixins.LLMValidated.validate_llm_rules", return_value=None)
    def test_valid_instance(self, mock_validate):
        code = (
            "from chat_workflow import atomic_workflow\n"
            "from pydantic import BaseModel, Field\n\n"
            "class MyModel(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Name")\n'
            "\n"
            "    @atomic_workflow\n"
            "    @classmethod\n"
            "    def create(cls, context: str):\n"
            '        """Create."""\n'
            "        ...\n"
        )
        model = GeneratedComponent(code=code)
        self.assertEqual(model.code, code)

    def test_missing_code(self):
        with self.assertRaises(ValidationError):
            GeneratedComponent()


if __name__ == "__main__":
    unittest.main(verbosity=2)
