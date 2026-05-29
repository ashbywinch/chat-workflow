"""Tests for create composite workflow."""
import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from workflows.workflow.models import (
    ComponentRequirement,
    GapAnalysis,
    Input,
    Output,
    ProcessAnalysis,
)
from workflows.workflow.workflow import Workflow, create


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class TestWorkflowCreate(unittest.TestCase):
    def _make_session(self) -> Session:
        return Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def test_has_workflow_attribute(self):
        self.assertTrue(
            getattr(create, "_is_workflow", False)
        )

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            create(process_description="test")
        self.assertIn("session", str(ctx.exception))

    @patch("workflows.workflow.models.process_analysis.ProcessAnalysis.generate_from_chat")
    @patch("workflows.workflow.models.input.Input.generate_from_chat")
    @patch("workflows.workflow.models.output.Output.generate_from_chat")
    @patch("workflows.workflow.workflow._resolve_gaps")
    @patch.object(Workflow, "_create_diagram")
    @patch("workflows.workflow.component.Component")
    def test_full_orchestration(
        self,
        mock_component_class,
        mock_create_diagram,
        mock_resolve_gaps,
        mock_collect_outputs,
        mock_collect_inputs,
        mock_analyze,
    ):
        """All sub-steps should be called in order."""
        analysis = ProcessAnalysis(
            phases=["Test"],
            activities=["Test"],
            orchestrating_component="Test",
            participants=["Test"],
        )
        mock_analyze.return_value = analysis

        inp = Input(
            source="Test",
            format="JSON",
            trigger_conditions="Test",
            dependencies=[],
            validation_criteria="Test",
        )
        mock_collect_inputs.return_value = [inp]

        out = Output(
            consumer="Test",
            format="JSON",
            success_criteria="Test",
            integration_points="Test",
            storage_requirements="Test",
        )
        mock_collect_outputs.return_value = [out]

        req = ComponentRequirement(
            name="TestComponent",
            purpose="Test",
            required_inputs=["Test"],
            expected_outputs=["Test"],
            component_type="artifact_producing",
        )
        mock_resolve_gaps.return_value = ([req], GapAnalysis(
            missing_components=[],
            missing_playbooks=[],
            integration_gaps=[],
            organizational_gaps=[],
            recommendations=[],
        ))

        workflow = Workflow(
            name="Test Workflow",
            diagram="sequenceDiagram",
            inputs=[inp],
            outputs=[out],
            components=[req],
            gap_analysis=None,
            architectural_validation="OK",
        )
        mock_create_diagram.return_value = workflow

        mock_component_instance = MagicMock()
        mock_component_instance.code_path = "/tmp/test.py"
        mock_component_class.create.return_value = mock_component_instance

        session = self._make_session()
        result = create(
            process_description="test process",
            session=session,
        )

        self.assertIsInstance(result, Workflow)
        self.assertEqual(result.name, "Test Workflow")
        mock_analyze.assert_called_once()
        mock_collect_inputs.assert_called_once()
        mock_collect_outputs.assert_called_once()
        mock_resolve_gaps.assert_called_once()
        mock_create_diagram.assert_called_once()
        self.assertEqual(mock_component_class.create.call_count, 1)

    @patch("workflows.workflow.models.process_analysis.ProcessAnalysis.generate_from_chat")
    @patch("workflows.workflow.models.input.Input.generate_from_chat")
    @patch("workflows.workflow.models.output.Output.generate_from_chat")
    @patch("workflows.workflow.workflow._resolve_gaps")
    @patch.object(Workflow, "_create_diagram")
    @patch("workflows.workflow.component.Component")
    def test_handles_multiple_components(
        self,
        mock_component_class,
        mock_create_diagram,
        mock_resolve_gaps,
        mock_collect_outputs,
        mock_collect_inputs,
        mock_analyze,
    ):
        """When multiple components identified, Component.create is called for each."""
        analysis = ProcessAnalysis(
            phases=["Test"],
            activities=["Test"],
            orchestrating_component="Test",
            participants=["Test"],
        )
        mock_analyze.return_value = analysis

        inp = Input(
            source="Test",
            format="JSON",
            trigger_conditions="Test",
            dependencies=[],
            validation_criteria="Test",
        )
        mock_collect_inputs.return_value = [inp]

        out = Output(
            consumer="Test",
            format="JSON",
            success_criteria="Test",
            integration_points="Test",
            storage_requirements="Test",
        )
        mock_collect_outputs.return_value = [out]

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
                component_type="value_stream",
            ),
        ]
        mock_resolve_gaps.return_value = (reqs, GapAnalysis(
            missing_components=[],
            missing_playbooks=[],
            integration_gaps=[],
            organizational_gaps=[],
            recommendations=[],
        ))

        workflow = Workflow(
            name="Test",
            diagram="seq",
            inputs=[inp],
            outputs=[out],
            components=reqs,
            gap_analysis=None,
            architectural_validation="OK",
        )
        mock_create_diagram.return_value = workflow

        mock_component_instance = MagicMock()
        mock_component_instance.code_path = "/tmp/test.py"
        mock_component_class.create.return_value = mock_component_instance

        session = self._make_session()
        result = create(
            process_description="multi-component process",
            session=session,
        )

        self.assertIsInstance(result, Workflow)
        self.assertEqual(mock_component_class.create.call_count, 2)

    @patch("workflows.workflow.models.process_analysis.ProcessAnalysis.generate_from_chat")
    @patch("workflows.workflow.models.input.Input.generate_from_chat")
    @patch("workflows.workflow.models.output.Output.generate_from_chat")
    @patch("workflows.workflow.workflow._resolve_gaps")
    @patch.object(Workflow, "_create_diagram")
    @patch("workflows.workflow.component.Component")
    def test_component_creation_failure_does_not_abort(
        self,
        mock_component_class,
        mock_create_diagram,
        mock_resolve_gaps,
        mock_collect_outputs,
        mock_collect_inputs,
        mock_analyze,
    ):
        """If one component creation fails, others still proceed."""
        analysis = ProcessAnalysis(
            phases=["Test"],
            activities=["Test"],
            orchestrating_component="Test",
            participants=["Test"],
        )
        mock_analyze.return_value = analysis

        inp = Input(
            source="Test",
            format="JSON",
            trigger_conditions="Test",
            dependencies=[],
            validation_criteria="Test",
        )
        mock_collect_inputs.return_value = [inp]

        out = Output(
            consumer="Test",
            format="JSON",
            success_criteria="Test",
            integration_points="Test",
            storage_requirements="Test",
        )
        mock_collect_outputs.return_value = [out]

        reqs = [
            ComponentRequirement(
                name="Good",
                purpose="Test",
                required_inputs=["A"],
                expected_outputs=["A"],
                component_type="artifact_producing",
            ),
            ComponentRequirement(
                name="Bad",
                purpose="Test",
                required_inputs=["B"],
                expected_outputs=["B"],
                component_type="value_stream",
            ),
            ComponentRequirement(
                name="AlsoGood",
                purpose="Test",
                required_inputs=["C"],
                expected_outputs=["C"],
                component_type="artifact_producing",
            ),
        ]
        mock_resolve_gaps.return_value = (reqs, GapAnalysis(
            missing_components=[],
            missing_playbooks=[],
            integration_gaps=[],
            organizational_gaps=[],
            recommendations=[],
        ))

        workflow = Workflow(
            name="Test",
            diagram="seq",
            inputs=[inp],
            outputs=[out],
            components=reqs,
            gap_analysis=None,
            architectural_validation="OK",
        )
        mock_create_diagram.return_value = workflow

        mock_component_class.create.side_effect = [
            MagicMock(code_path="/tmp/good.py"),
            RuntimeError("Failed!"),
            MagicMock(code_path="/tmp/also_good.py"),
        ]

        session = self._make_session()
        result = create(
            process_description="mixed component process",
            session=session,
        )

        self.assertIsInstance(result, Workflow)
        self.assertEqual(mock_component_class.create.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)