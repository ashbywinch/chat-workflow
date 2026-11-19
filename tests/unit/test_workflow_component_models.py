"""Tests for Workflow and Component models."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from chat_workflow.mixins import get_blob_fields
from workflows.workflow.component import Component
from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.models import (
    Deliverable,
    GapAnalysis,
    Resource,
)
from workflows.workflow.workflow_spec import WorkflowSpec


class TestWorkflowModel(unittest.TestCase):
    def make_valid_workflow(self) -> WorkflowSpec:
        return WorkflowSpec.model_construct(
            name="Order Processing Workflow",
            diagram=(
                "sequenceDiagram\n"
                "participant Order Management: Order Processing\n"
                "participant Inventory System: Stock Check\n"
                "Order Management: Order Processing->>Inventory System: Stock Check: "
                "Check availability"
            ),
            inputs=[
                Resource(
                    source="Customer",
                    format="JSON",
                    validation_criteria="Must include items",
                )
            ],
            outputs=[
                Deliverable(
            name="Test",
            description="Test",
                    consumer="Inventory",
                    format="Event",
                    success_criteria="Items reserved",
                    integration_points="Inventory API",
                    storage_requirements="None",
                )
            ],
            components=[
                ComponentSpec(
                    name="Order",
                    purpose="Manage orders",
                    scope_description="Manage orders",
                    required_inputs=["Customer details"],
                    component_type="artifact_producing",
                    owner_role="Owner",
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
        blobs = get_blob_fields(WorkflowSpec)
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
        rules = WorkflowSpec.collect_all_rules()
        # Should include per-field Validation rules AND per-model _validation_rules
        self.assertTrue(any("must be nouns" in r for r in rules))

    def test_participant_naming_is_deterministic(self):
        good = 'sequenceDiagram\n    participant NoteTaker as "Note Taker: Orchestration"'
        self.assertEqual(WorkflowSpec._validate_diagram_syntax(good), good)

        with self.assertRaises(ValueError):
            WorkflowSpec._validate_diagram_syntax(
                'sequenceDiagram\n    participant You as "You: Orchestration"'
            )
        with self.assertRaises(ValueError):
            WorkflowSpec._validate_diagram_syntax(
                'sequenceDiagram\n    participant A as "Minutes: Capture the whole meeting"'
            )

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            WorkflowSpec()

    def test_gap_analysis_optional(self):
        wf = self.make_valid_workflow()
        wf.gap_analysis = None
        self.assertIsNone(wf.gap_analysis)


class TestComponentModel(unittest.TestCase):
    def make_valid_component(self) -> Component:
        return Component.model_construct(
            name="Order",
            purpose="Manage the complete order lifecycle",
            code_path=Path("/tmp/workflows/order/order.py"),
            model_class="Order",
            expert_role="Order Management Expert",
            component_type="artifact_producing",
            owner_role="Owner",
        )

    def test_valid_component(self):
        c = self.make_valid_component()
        self.assertEqual(c.name, "Order")
        self.assertEqual(c.code_path, Path("/tmp/workflows/order/order.py"))

    def test_default_execution_complexity(self):
        c = self.make_valid_component()
        self.assertEqual(c.execution_complexity, "simple")

    def test_custom_execution_complexity(self):
        c = Component.model_construct(
            name="Invoice",
            purpose="Handle invoicing",
            code_path=Path("/tmp/invoice.py"),
            model_class="Invoice",
            expert_role="Billing Expert",
            component_type="artifact_producing",
            owner_role="Owner",
            execution_complexity="complex",
        )
        self.assertEqual(c.execution_complexity, "complex")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            Component()


if __name__ == "__main__":
    unittest.main(verbosity=2)
