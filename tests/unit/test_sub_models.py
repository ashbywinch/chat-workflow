"""Tests for workflow sub-models."""

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from workflows.workflow import ComponentSourceCode
from workflows.workflow.models import (
    ComponentResponsibilities,
    Deliverable,
    GapAnalysis,
    ProcessDefinition,
    Resource,
)
from workflows.workflow.models.gap_analysis import IntegrationGap, OwnershipGap


class TestProcessDefinition(unittest.TestCase):
    def test_valid_instance(self):
        model = ProcessDefinition(
            phases=["Intake", "Review"],
            activities=["Receive request", "Check completeness"],
            orchestrating_component="Case Management",
            participants=["Case Worker", "Manager"],
        )
        self.assertEqual(len(model.phases), 2)
        self.assertEqual(model.orchestrating_component, "Case Management")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ProcessDefinition()


class TestResource(unittest.TestCase):
    def test_valid_instance(self):
        model = Resource(
            source="Customer",
            format="JSON",
            trigger_conditions="New request submitted",
            dependencies=["Customer profile"],
            validation_criteria="Must include customer ID",
        )
        self.assertEqual(model.source, "Customer")
        self.assertEqual(model.dependencies, ["Customer profile"])

    def test_default_dependencies(self):
        model = Resource(
            source="System",
            format="CSV",
            trigger_conditions="Daily batch",
            validation_criteria="File must exist",
        )
        self.assertEqual(model.dependencies, [])

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Resource()


class TestDeliverable(unittest.TestCase):
    def test_valid_instance(self):
        model = Deliverable(
            name="Invoice",
            description="Processed invoice records",
            consumer="Billing System",
            format="XML",
            success_criteria="All invoices processed",
            integration_points="API endpoint",
            storage_requirements="Database archive",
        )
        self.assertEqual(model.consumer, "Billing System")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Deliverable()


class TestGapAnalysis(unittest.TestCase):
    def test_valid_instance(self):
        model = GapAnalysis(
            missing_components=["Payment Processor"],
            missing_playbooks=["Create Payment Playbook"],
            integration_gaps=[IntegrationGap(
                between="Billing → Payment",
                description="Handoff unclear",
            )],
            organizational_gaps=[OwnershipGap(
                activity="Reconciliation",
                reason="No owner assigned",
            )],
            recommendations=["Create Payment Processor component"],
        )
        self.assertEqual(len(model.missing_components), 1)

    def test_empty_gap_analysis_is_valid(self):
        model = GapAnalysis()
        self.assertEqual(len(model.missing_components), 0)


class TestComponentResponsibilities(unittest.TestCase):
    def test_valid_instance(self):
        model = ComponentResponsibilities(
            name="Invoice",
            purpose="Manage invoice lifecycle",
            required_inputs=["Order details"],
            scope_description="description", 
            component_type="artifact_producing",
        )
        self.assertEqual(model.name, "Invoice")
        self.assertEqual(model.component_type, "artifact_producing")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ComponentResponsibilities()


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
        model = ComponentSourceCode.model_construct(code=code)
        self.assertEqual(model.code, code)

    def test_missing_code(self):
        with self.assertRaises(ValidationError):
            ComponentSourceCode()


if __name__ == "__main__":
    unittest.main(verbosity=2)
