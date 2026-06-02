"""Integration test: Workflow.create orchestrates Component.create end-to-end."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.models import AgentIntent, AgentResponse
from workflows.workflow import GeneratedComponent, Workflow
from workflows.workflow.models import (
    ComponentRequirement,
    Deliverable,
    GapAnalysis,
    ProcessDefinition,
    Resource,
)


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class TestWorkflowIntegration(unittest.TestCase):
    """End-to-end integration test: Workflow.create() → Component.create() → files."""

    def setUp(self):
        self.session = Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_end_to_end_flow(self, mock_call_llm):
        """Full flow: process in → Workflow + Component files out."""
        analysis = ProcessDefinition(
            phases=["Intake", "Process"],
            activities=["Receive", "Fulfill"],
            orchestrating_component="Order Management",
            participants=["Customer", "System"],
        )

        inp = Resource(
            source="Customer",
            format="JSON",
            trigger_conditions="Order placed",
            dependencies=[],
            validation_criteria="Must have items",
        )

        out = Deliverable(
            consumer="Inventory",
            format="Event",
            success_criteria="Items reserved",
            integration_points="API",
            storage_requirements="DB",
        )

        req = ComponentRequirement(
            name="Order",
            purpose="Manage orders",
            required_inputs=["Details"],
            expected_outputs=["Confirmation"],
            component_type="artifact_producing",
        )

        generated_code = (
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, Field\n\n"
            "class Order(BaseModel):\n"
            '    name: str = Field(..., description="Order name")\n'
        )

        workflow = Workflow(
            name="Order Processing Workflow",
            diagram="sequenceDiagram\nparticipant A\nA->>B: hello",
            inputs=[inp],
            outputs=[out],
            components=[req],
            gap_analysis=None,
            architectural_validation="OK",
        )

        mock_responses = [
            AgentResponse[list[Deliverable]](intent=AgentIntent.SUCCESS, result=[out]),
            AgentResponse[list[Resource]](intent=AgentIntent.SUCCESS, result=[inp]),
            AgentResponse[str](intent=AgentIntent.SUCCESS, result="raw notes"),
            AgentResponse[ProcessDefinition](intent=AgentIntent.SUCCESS, result=analysis),
            AgentResponse[list[ComponentRequirement]](intent=AgentIntent.SUCCESS, result=[req]),
            AgentResponse[GapAnalysis](
                intent=AgentIntent.SUCCESS,
                result=GapAnalysis(
                    missing_components=[],
                    missing_playbooks=[],
                    integration_gaps=[],
                    organizational_gaps=[],
                    recommendations=[],
                ),
            ),
            AgentResponse[Workflow](intent=AgentIntent.SUCCESS, result=workflow),
            AgentResponse[Workflow](intent=AgentIntent.SUCCESS, result=workflow),
            AgentResponse[GeneratedComponent](
                intent=AgentIntent.SUCCESS,
                result=GeneratedComponent(code=generated_code),
            ),
        ]
        mock_call_llm.side_effect = mock_responses

        with (
            patch("workflows.workflow.component.verify_code") as mock_verify,
            TemporaryDirectory() as tmpdir,
            patch.object(Path, "cwd", return_value=Path(tmpdir)),
        ):
            mock_verify.return_value = generated_code

            result = Workflow.create(
                process_description="Order processing",
                session=self.session,
            )

            self.assertIsInstance(result, Workflow)
            self.assertEqual(result.name, "Order Processing Workflow")
            self.assertIn("sequenceDiagram", result.diagram)
            self.assertEqual(len(result.components), 1)
            self.assertEqual(result.components[0].name, "Order")
            self.assertEqual(mock_call_llm.call_count, 8)

            expected_path = Path(tmpdir) / "workflows" / "order" / "order.py"
            self.assertTrue(expected_path.exists(), f"Component file not found at {expected_path}")
            file_content = expected_path.read_text()
            self.assertIn("class Order(BaseModel)", file_content)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_handles_multiple_components(self, mock_call_llm):
        """When multiple components identified, files are written for each."""
        analysis = ProcessDefinition(
            phases=["Test"],
            activities=["Test"],
            orchestrating_component="Test",
            participants=["Test"],
        )

        inp = Resource(
            source="Test",
            format="JSON",
            trigger_conditions="Test",
            dependencies=[],
            validation_criteria="Test",
        )

        out = Deliverable(
            consumer="Test",
            format="JSON",
            success_criteria="Test",
            integration_points="Test",
            storage_requirements="Test",
        )

        reqs = [
            ComponentRequirement(
                name="ComponentA",
                purpose="Test A",
                required_inputs=["A"],
                expected_outputs=["A"],
                component_type="artifact_producing",
            ),
            ComponentRequirement(
                name="ComponentB",
                purpose="Test B",
                required_inputs=["B"],
                expected_outputs=["B"],
                component_type="artifact_producing",
            ),
        ]

        workflow = Workflow(
            name="Multi Workflow",
            diagram="seq",
            inputs=[inp],
            outputs=[out],
            components=reqs,
            gap_analysis=None,
            architectural_validation="OK",
        )

        code_a = "class ComponentA(BaseModel):\n    pass\n"
        code_b = "class ComponentB(BaseModel):\n    pass\n"

        mock_responses = [
            AgentResponse[list[Deliverable]](intent=AgentIntent.SUCCESS, result=[out]),
            AgentResponse[list[Resource]](intent=AgentIntent.SUCCESS, result=[inp]),
            AgentResponse[str](intent=AgentIntent.SUCCESS, result="raw notes"),
            AgentResponse[ProcessDefinition](intent=AgentIntent.SUCCESS, result=analysis),
            AgentResponse[list[ComponentRequirement]](intent=AgentIntent.SUCCESS, result=reqs),
            AgentResponse[GapAnalysis](
                intent=AgentIntent.SUCCESS,
                result=GapAnalysis(
                    missing_components=[],
                    missing_playbooks=[],
                    integration_gaps=[],
                    organizational_gaps=[],
                    recommendations=[],
                ),
            ),
            AgentResponse[Workflow](intent=AgentIntent.SUCCESS, result=workflow),
            AgentResponse[Workflow](intent=AgentIntent.SUCCESS, result=workflow),
            AgentResponse[GeneratedComponent](intent=AgentIntent.SUCCESS, result=GeneratedComponent(code=code_a)),
            AgentResponse[GeneratedComponent](intent=AgentIntent.SUCCESS, result=GeneratedComponent(code=code_b)),
        ]
        mock_call_llm.side_effect = mock_responses

        with (
            patch("workflows.workflow.component.verify_code") as mock_verify,
            TemporaryDirectory() as tmpdir,
            patch.object(Path, "cwd", return_value=Path(tmpdir)),
        ):
            mock_verify.side_effect = [code_a, code_b]

            result = Workflow.create(
                process_description="multi-component test",
                session=self.session,
            )

            self.assertIsInstance(result, Workflow)
            self.assertEqual(len(result.components), 2)

            path_a = Path(tmpdir) / "workflows" / "componenta" / "componenta.py"
            path_b = Path(tmpdir) / "workflows" / "componentb" / "componentb.py"
            self.assertTrue(path_a.exists())
            self.assertTrue(path_b.exists())
            self.assertIn("class ComponentA", path_a.read_text())
            self.assertIn("class ComponentB", path_b.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
