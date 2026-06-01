"""Tests for Workflow and Component models."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from chat_workflow.mixins import get_blob_fields
from workflows.workflow.component import Component
from workflows.workflow.component_responsibilities import ComponentResponsibilities
from workflows.workflow.models import (
    GapAnalysis,
    Input,
    Output,
)
from workflows.workflow.workflow import Workflow


class TestWorkflowModel(unittest.TestCase):
    def make_valid_workflow(self) -> Workflow:
        return Workflow.model_construct(
            name="Order Processing Workflow",
            diagram=(
                "sequenceDiagram\n"
                "participant Order Management: Order Processing\n"
                "participant Inventory System: Stock Check\n"
                "Order Management: Order Processing->>Inventory System: Stock Check: "
                "Check availability"
            ),
            inputs=[
                Input(
                    source="Customer",
                    format="JSON",
                    trigger_conditions="Order placed",
                    validation_criteria="Must include items",
                )
            ],
            outputs=[
                Output(
                    consumer="Inventory",
                    format="Event",
                    success_criteria="Items reserved",
                    integration_points="Inventory API",
                    storage_requirements="None",
                )
            ],
            components=[
                ComponentResponsibilities(
                    name="Order",
                    purpose="Manage orders",
                    scope_description="Manage orders",
                    required_inputs=["Customer details"],
                    component_type="artifact_producing",
                )
            ],
            gap_analysis=GapAnalysis(
                missing_components=[],
                missing_playbooks=[],
                integration_gaps=[],
                organizational_gaps=[],
                recommendations=[],
            ),
            architectural_validation="All components properly owned",
        )

    def test_valid_workflow(self):
        wf = self.make_valid_workflow()
        self.assertEqual(wf.name, "Order Processing Workflow")
        self.assertEqual(len(wf.inputs), 1)
        self.assertEqual(len(wf.components), 1)

    def test_get_blob_fields_finds_diagram(self):
        blobs = get_blob_fields(Workflow)
        self.assertIn("diagram", blobs)
        self.assertEqual(blobs["diagram"], ".mmd")

    def test_materialize_blobs(self):
        wf = self.make_valid_workflow()
        with TemporaryDirectory() as tmpdir:
            wf.materialize_blobs(Path(tmpdir))
            path = wf.get_blob_path("diagram")
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".mmd")

    def test_validation_rules_collected(self):
        rules = Workflow.collect_all_rules()
        # Should include per-field Validation rules AND per-model _validation_rules
        self.assertTrue(any("sequenceDiagram" in r for r in rules))
        self.assertTrue(any("artifact-based naming" in r for r in rules))

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Workflow()

    def test_gap_analysis_optional(self):
        wf = self.make_valid_workflow()
        wf.gap_analysis = None
        self.assertIsNone(wf.gap_analysis)


class TestComponentModel(unittest.TestCase):
    def make_valid_component(self) -> Component:
        return Component(
            name="Order",
            purpose="Manage the complete order lifecycle",
            code_path=Path("/tmp/workflows/order/order.py"),
            model_class="Order",
            expert_role="Order Management Expert",
            component_type="artifact_producing",
        )

    def test_valid_component(self):
        c = self.make_valid_component()
        self.assertEqual(c.name, "Order")
        self.assertEqual(c.code_path, Path("/tmp/workflows/order/order.py"))

    def test_default_execution_complexity(self):
        c = self.make_valid_component()
        self.assertEqual(c.execution_complexity, "simple")

    def test_custom_execution_complexity(self):
        c = Component(
            name="Invoice",
            purpose="Handle invoicing",
            code_path=Path("/tmp/invoice.py"),
            model_class="Invoice",
            expert_role="Billing Expert",
            component_type="artifact_producing",
            execution_complexity="complex",
        )
        self.assertEqual(c.execution_complexity, "complex")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Component()


if __name__ == "__main__":
    unittest.main(verbosity=2)
