"""Tests for ComponentResponsibilities.identify_from_chat, GapAnalysis.analyze_from_chat,
and _resolve_gaps."""

import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.models import AgentIntent, AgentResponse
from workflows.workflow.models import (
    ComponentResponsibilities,
    Deliverable,
    GapAnalysis,
    ProcessDefinition,
    Resource,
)
from workflows.workflow.models.gap_analysis import IntegrationGap
from workflows.workflow.workflow import _resolve_gaps


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


class TestIdentifyComponents(unittest.TestCase):
    def test_has_workflow_attribute(self):
        self.assertTrue(getattr(ComponentResponsibilities.identify_from_chat, "_is_workflow", False))

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            ComponentResponsibilities.identify_from_chat(
                analysis=_make_analysis(),
                inputs=_make_inputs(),
                outputs=_make_outputs(),
            )
        self.assertIn("session", str(ctx.exception))

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_returns_component_requirements(self, mock_call_llm):
        expected = [
            ComponentResponsibilities(
                name="Order",
                purpose="Manage orders",
                required_inputs=["Customer details"],
                scope_description="description", 
                component_type="artifact_producing",
            )
        ]
        mock_call_llm.return_value = AgentResponse[list[ComponentResponsibilities]](
            intent=AgentIntent.SUCCESS,
            result=expected,
        )
        session = _make_session()
        result = ComponentResponsibilities.identify_from_chat(
            analysis=_make_analysis(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ComponentResponsibilities)


class TestAnalyzeGaps(unittest.TestCase):
    def test_has_workflow_attribute(self):
        self.assertTrue(getattr(GapAnalysis.analyze_from_chat, "_is_workflow", False))

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            GapAnalysis.analyze_from_chat(
                components=[],
                analysis=_make_analysis(),
            )
        self.assertIn("session", str(ctx.exception))

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_returns_gap_analysis(self, mock_call_llm):
        expected = GapAnalysis(
            missing_components=[],
            missing_playbooks=[],
            integration_gaps=[],
            organizational_gaps=[],
            recommendations=[],
        )
        mock_call_llm.return_value = AgentResponse[GapAnalysis](
            intent=AgentIntent.SUCCESS,
            result=expected,
        )
        session = _make_session()
        result = GapAnalysis.analyze_from_chat(
            components=[],
            analysis=_make_analysis(),
            session=session,
        )
        self.assertIsInstance(result, GapAnalysis)


class TestResolveGaps(unittest.TestCase):
    @patch("workflows.workflow.workflow._GapAnalysis.analyze_from_chat")
    @patch("workflows.workflow.workflow.ComponentResponsibilities.identify_from_chat")
    def test_loop_terminates_when_clean(self, mock_identify, mock_analyze):
        """When gaps are clean on first try, loop runs once."""
        component = ComponentResponsibilities(
            name="Order",
            purpose="Manage orders",
            required_inputs=["Details"],
            scope_description="description", 
            component_type="artifact_producing",
        )
        mock_identify.return_value = [component]
        mock_analyze.return_value = GapAnalysis(
            missing_components=[],
            missing_playbooks=[],
            integration_gaps=[],
            organizational_gaps=[],
            recommendations=[],
        )

        session = _make_session()
        session.io.prompt.return_value = ""
        components, gaps = _resolve_gaps(
            analysis=_make_analysis(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )

        self.assertEqual(len(components), 1)
        self.assertIsInstance(components[0], ComponentResponsibilities)
        self.assertEqual(components[0].name, "Order")
        self.assertEqual(components[0].scope_description, "Manage orders")
        self.assertEqual(len(gaps.missing_components), 0)
        mock_identify.assert_called_once()
        mock_analyze.assert_called_once()

    @patch("workflows.workflow.workflow._GapAnalysis.analyze_from_chat")
    @patch("workflows.workflow.workflow.ComponentResponsibilities.identify_from_chat")
    def test_loop_retries_on_gaps(self, mock_identify, mock_analyze):
        """When gaps exist, loop retries until clean."""
        component = ComponentResponsibilities(
            name="Order",
            purpose="Manage orders",
            required_inputs=["Details"],
            scope_description="description", 
            component_type="artifact_producing",
        )
        mock_identify.return_value = [component]

        mock_analyze.side_effect = [
            GapAnalysis(
                missing_components=["Payment"],
                missing_playbooks=[],
                integration_gaps=[IntegrationGap(
                    between="Order → Payment",
                    description="Handoff unclear",
                )],
                organizational_gaps=[],
                recommendations=["Create Payment component"],
            ),
            GapAnalysis(
                missing_components=[],
                missing_playbooks=[],
                integration_gaps=[],
                organizational_gaps=[],
                recommendations=[],
            ),
        ]

        session = _make_session()
        session.io.prompt.return_value = ""
        components, gaps = _resolve_gaps(
            analysis=_make_analysis(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )

        self.assertEqual(len(components), 1)
        self.assertIsInstance(components[0], ComponentResponsibilities)
        self.assertEqual(len(gaps.missing_components), 0)
        self.assertEqual(mock_identify.call_count, 2)
        self.assertEqual(mock_analyze.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
