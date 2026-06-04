"""Tests for Workflow._generate_diagram and _create_diagram."""

import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.models import AgentIntent, AgentResponse
from workflows.workflow.component_responsibilities import ComponentResponsibilities
from workflows.workflow.models import (
    Deliverable,
    ProcessDefinition,
    Resource,
)
from workflows.workflow.workflow import Workflow


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False
    model_supports_tools = False
    api_base = None
    api_key_env = None


def _make_session() -> Session:
    return Session(
        io=MagicMock(),
        state=SessionLog(),
        config=FakeConfig(),
    )


def _make_analysis() -> ProcessDefinition:
    return ProcessDefinition(
        phases=["Intake", "Process", "Complete"],
        activities=["Receive", "Validate", "Finalize"],
        orchestrating_component="Order Management",
        participants=["Customer", "System"],
    )


def _make_components() -> list[ComponentResponsibilities]:
    return [
        ComponentResponsibilities(
            name="Order",
            purpose="Manage orders",
            scope_description="Manage orders",
            required_inputs=["Details"],
            component_type="artifact_producing",
        )
    ]


def _make_inputs() -> list[Resource]:
    return [
        Resource(
            source="Customer",
            format="JSON",
            trigger_conditions="Order placed",
            validation_criteria="Must have items",
        )
    ]


def _make_outputs() -> list[Deliverable]:
    return [
        Deliverable(
            name="Test",
            description="Test",
            consumer="Inventory",
            format="Event",
            success_criteria="Items reserved",
            integration_points="API",
            storage_requirements="DB",
        )
    ]


class TestGenerateDiagram(unittest.TestCase):
    def test_has_workflow_attribute(self):
        self.assertTrue(getattr(Workflow._generate_diagram, "_is_workflow", False))

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            Workflow._generate_diagram(
                analysis=_make_analysis(),
                components=_make_components(),
                inputs=_make_inputs(),
                outputs=_make_outputs(),
            )
        self.assertIn("session", str(ctx.exception))

    @patch.object(Workflow, "collect_all_rules", return_value=[])
    @patch.object(AtomicWorkflow, "_call_llm")
    def test_returns_workflow(self, mock_call_llm, mock_collect):
        expected = Workflow.model_construct(
            name="Test Workflow",
            diagram="sequenceDiagram\nparticipant A\nA->>B: hello",
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            components=_make_components(),
            gap_analysis=None,
            architectural_validation="All components properly owned",
        )
        mock_call_llm.return_value = AgentResponse[Workflow].model_construct(
            intent=AgentIntent.SUCCESS,
            result=expected,
        )
        session = _make_session()
        result = Workflow._generate_diagram(
            analysis=_make_analysis(),
            components=_make_components(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )
        self.assertIsInstance(result, Workflow)
        self.assertEqual(result.name, "Test Workflow")
        self.assertIn("sequenceDiagram", result.diagram)


class TestCreateDiagram(unittest.TestCase):
    @patch("workflows.evaluation_criteria.refine.refine")
    @patch.object(Workflow, "_generate_diagram")
    def test_materializes_blobs(self, mock_generate, mock_refine):
        workflow = Workflow.model_construct(
            name="Test Workflow",
            diagram="sequenceDiagram\nA->>B: test",
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            components=_make_components(),
            gap_analysis=None,
            architectural_validation="OK",
        )
        mock_generate.return_value = workflow
        mock_refine.return_value = workflow

        session = _make_session()
        result = Workflow._create_diagram(
            analysis=_make_analysis(),
            components=_make_components(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )

        self.assertIsInstance(result, Workflow)
        mock_generate.assert_called_once()
        mock_refine.assert_called_once()

    @patch("workflows.evaluation_criteria.refine.refine")
    @patch.object(Workflow, "_generate_diagram")
    def test_refine_loop_reruns_on_changes(self, mock_generate, mock_refine):
        original = Workflow.model_construct(
            name="Test Workflow",
            diagram="sequenceDiagram\nA->>B: original",
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            components=_make_components(),
            gap_analysis=None,
            architectural_validation="OK",
        )
        changed = Workflow.model_construct(
            name="Test Workflow",
            diagram="sequenceDiagram\nA->>B: changed",
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            components=_make_components(),
            gap_analysis=None,
            architectural_validation="OK",
        )
        mock_generate.return_value = original
        mock_refine.side_effect = [changed, changed]

        session = _make_session()
        result = Workflow._create_diagram(
            analysis=_make_analysis(),
            components=_make_components(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
            max_refinements=5,
        )

        self.assertEqual(result.diagram, "sequenceDiagram\nA->>B: changed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
