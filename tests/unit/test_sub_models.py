"""Tests for workflow sub-models."""

import unittest

from pydantic import ValidationError

from workflows.workflow import ComponentSourceCode
from workflows.workflow.models import (
    ComponentSpec,
    Deliverable,
    GapAnalysis,
    ProcessDefinition,
    ProcessStage,
    Resource,
)
from workflows.workflow.models.gap_analysis import IntegrationGap, OwnershipGap


class TestProcessDefinition(unittest.TestCase):
    def test_valid_instance(self):
        model = ProcessDefinition(
            stages=[
                ProcessStage(name="Intake", steps=["Receive request"]),
                ProcessStage(name="Review", steps=["Check completeness"]),
            ],
            orchestrating_component="Case Management",
            participants=["Case Worker", "Manager"],
        )
        self.assertEqual(len(model.phases), 2)
        self.assertEqual(model.orchestrating_component, "Case Management")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ProcessDefinition()

    def test_a_stage_that_repeats_its_step_says_how_to_repair(self):
        """The rejection is a repair instruction, not just a complaint."""
        with self.assertRaises(ValueError) as context:
            ProcessDefinition(
                stages=[ProcessStage(name="Collect receipts", steps=["collect receipts"])],
                orchestrating_component="You",
                participants=["You"],
            )

        message = str(context.exception)
        self.assertIn("rename the stage", message)
        self.assertIn("drop the duplicate step", message)


class TestResource(unittest.TestCase):
    def test_valid_instance(self):
        model = Resource(
            source="Customer",
            format="JSON",
            dependencies=["Customer profile"],
            validation_criteria="Must include customer ID",
        )
        self.assertEqual(model.source, "Customer")
        self.assertEqual(model.dependencies, ["Customer profile"])

    def test_default_dependencies(self):
        model = Resource(
            source="System",
            format="CSV",
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

    def test_consumer_must_not_be_the_artifact(self):
        with self.assertRaises(ValidationError):
            Deliverable(
                name="Customer",
                description="Customer profile",
                consumer="Customer",
                format="JSON",
                success_criteria="Complete",
                integration_points="API",
                storage_requirements="Database",
            )


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


class TestComponentSpec(unittest.TestCase):
    def test_valid_instance(self):
        model = ComponentSpec(
            name="Invoice",
            purpose="Manage invoice lifecycle",
            required_inputs=["Order details"],
            scope_description="description", 
            component_type="artifact_producing",
            owner_role="Owner",
        )
        self.assertEqual(model.name, "Invoice")
        self.assertEqual(model.component_type, "artifact_producing")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ComponentSpec()

    def test_name_ending_in_activity_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            ComponentSpec(
                name="Minutes Distribution",
                purpose="Distribute the minutes",
                required_inputs=["Final Minutes"],
                scope_description=(
                    "Sends the minutes to attendees. Out of scope: editing them."
                ),
                component_type="artifact_producing",
                owner_role="Owner",
            )
        self.assertIn("Minutes Distribution Record", str(ctx.exception))


class TestComponentSourceCode(unittest.TestCase):
    def test_valid_instance(self):
        code = (
            "from chat_workflow import atomic_workflow_step\n"
            "from pydantic import BaseModel, Field\n\n"
            "class MyModel(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Name")\n'
            "\n"
            "    @atomic_workflow_step\n"
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
