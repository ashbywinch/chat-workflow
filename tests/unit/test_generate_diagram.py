"""Tests for WorkflowSpec._generate_diagram and _create_diagram."""

import unittest
from typing import override
from unittest.mock import MagicMock, patch

from pyfakefs.fake_filesystem_unittest import TestCase

from chat_workflow import Session, SessionLog
from chat_workflow.atomic_workflow_step import AtomicWorkflowStep
from chat_workflow.exceptions import ValidationError
from chat_workflow.models import AgentIntent, AgentResponse
from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.models import (
    Deliverable,
    ProcessDefinition,
    Resource,
)
from workflows.workflow.workflow_spec import (
    WorkflowSpec,
    _validate_diagram_structure,
)


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


def _make_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            name="Order",
            purpose="Manage orders",
            scope_description="Manage orders",
            required_inputs=["Details"],
            component_type="artifact_producing",
            owner_role="Owner",
        ),
        ComponentSpec(
            name="Warehouse",
            purpose="Store orders",
            scope_description="Store orders",
            required_inputs=["Items"],
            component_type="artifact_producing",
            owner_role="Owner",
        ),
    ]


def _make_inputs() -> list[Resource]:
    return [
        Resource(
            source="Customer",
            format="JSON",
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
        self.assertTrue(getattr(WorkflowSpec._generate_diagram, "_is_workflow_step", False))

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            WorkflowSpec._generate_diagram(
                analysis=_make_analysis(),
                components=_make_components(),
                inputs=_make_inputs(),
                outputs=_make_outputs(),
            )
        self.assertIn("session", str(ctx.exception))

    @patch.object(WorkflowSpec, "collect_all_rules", return_value=[])
    @patch.object(AtomicWorkflowStep, "_call_llm")
    def test_returns_workflow(self, mock_call_llm, mock_collect):
        expected = WorkflowSpec.model_construct(
            name="Test Workflow",
            diagram="sequenceDiagram\nparticipant A\nA->>B: hello",
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            components=_make_components(),
            gap_analysis=None,
            architectural_validation="All components properly owned",
        )
        mock_call_llm.return_value = AgentResponse[WorkflowSpec].model_construct(
            intent=AgentIntent.SUCCESS,
            result=expected,
        )
        session = _make_session()
        result = WorkflowSpec._generate_diagram(
            analysis=_make_analysis(),
            components=_make_components(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )
        self.assertIsInstance(result, WorkflowSpec)
        self.assertEqual(result.name, "Test Workflow")
        self.assertIn("sequenceDiagram", result.diagram)


class TestDiagramStructure(unittest.TestCase):
    """The diagram's mechanical contract, with the repair in the message."""

    def _check(self, diagram: str) -> None:
        _validate_diagram_structure(
            diagram, _make_components(), _make_outputs(), _make_inputs()
        )

    def test_a_complete_diagram_passes(self):
        self._check(
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant Inventory as "Inventory: Receive"\n'
            "Order->>Inventory: reservation\n"
            "Inventory-->>Order: confirmed\n"
            "alt item unavailable\n"
            "Order->>Inventory: backorder\n"
            "else available\n"
            "Order->>Inventory: confirm\n"
            "end\n"
            "Order->>Inventory: outcome"
        )

    def test_a_long_message_needs_a_break(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                'participant Inventory as "Inventory: Receive"\n'
                "Order->>Inventory: a message that is clearly far too long to fit on one line of the diagram\n"
                "Inventory-->>Order: ack\n"
                "opt may happen\n"
                "Order->>Inventory: more\n"
                "end\n"
                "Order->>Inventory: outcome"
            )

        self.assertIn("<br/>", str(ctx.exception))

    def test_a_missing_return_flow_names_the_repair(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                'participant Warehouse as "Warehouse: Store"\n'
                'participant Inventory as "Inventory: Receive"\n'
                "alt choice\n"
                "Order->>Warehouse: one\n"
                "else other\n"
                "Order->>Warehouse: two\n"
                "end\n"
                "Order->>Inventory: outcome"
            )

        self.assertIn("return flow", str(ctx.exception))

    def test_a_return_flow_after_the_outcome_names_the_order(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                'participant Inventory as "Inventory: Receive"\n'
                "Order->>Inventory: outcome\n"
                "Inventory-->>Order: ack\n"
                "opt may happen\n"
                "Order->>Inventory: more\n"
                "end\n"
                "Inventory-->>Order: ack again"
            )

        self.assertIn("before the outcome", str(ctx.exception))

    def test_a_self_loop_outcome_names_the_repair(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                'participant Inventory as "Inventory: Receive"\n'
                "Order->>Inventory: start\n"
                "Inventory-->>Order: ack\n"
                "opt may happen\n"
                "Order->>Inventory: more\n"
                "end\n"
                "Order->>Order: done"
            )

        self.assertIn("self-loop", str(ctx.exception))

    def test_the_outcome_goes_to_the_consumer(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                'participant Inventory as "Inventory: Receive"\n'
                "Order->>Inventory: start\n"
                "Inventory-->>Order: ack\n"
                "opt may happen\n"
                "Order->>Inventory: more\n"
                "end\n"
                "Inventory->>Order: done"
            )

        self.assertIn("consumer", str(ctx.exception))

    def test_a_missing_decision_point_names_the_repair(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                'participant Inventory as "Inventory: Receive"\n'
                "Order->>Inventory: start\n"
                "Inventory-->>Order: ack"
            )

        self.assertIn("decision point", str(ctx.exception))

    def test_an_undeclared_participant_names_it(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Order as "Order: Process"\n'
                "opt may happen\n"
                "Order->>Ghost: start\n"
                "end"
            )

        self.assertIn("Ghost", str(ctx.exception))
        self.assertIn("not declared", str(ctx.exception))

    def test_a_consumer_may_be_shortened(self):
        outputs = _make_outputs()
        outputs[0] = outputs[0].model_copy(
            update={"consumer": "Meeting attendees and absent invitees"}
        )
        _validate_diagram_structure(
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant Attendees as "Meeting attendees: Receive"\n'
            "Order->>Attendees: start\n"
            "Attendees-->>Order: ack\n"
            "opt may happen\n"
            "Order->>Attendees: more\n"
            "end\n"
            "Order->>Attendees: outcome",
            _make_components(),
            outputs,
            _make_inputs(),
        )

    def test_a_single_word_consumer_may_be_shortened(self):
        outputs = _make_outputs()
        outputs[0] = outputs[0].model_copy(update={"consumer": "My team"})
        _validate_diagram_structure(
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant Team as "Team: Receive"\n'
            "Order->>Team: start\n"
            "Team-->>Order: ack\n"
            "opt may happen\n"
            "Order->>Team: more\n"
            "end\n"
            "Order->>Team: outcome",
            _make_components(),
            outputs,
            _make_inputs(),
        )

    def test_a_consumer_with_punctuation_matches_its_prefix(self):
        outputs = _make_outputs()
        outputs[0] = outputs[0].model_copy(
            update={"consumer": "Meeting attendees, who use the action items"}
        )
        _validate_diagram_structure(
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant Attendees as "Meeting attendees: Deliver"\n'
            "Order->>Attendees: start\n"
            "Attendees-->>Order: ack\n"
            "opt may happen\n"
            "Order->>Attendees: more\n"
            "end\n"
            "Order->>Attendees: outcome",
            _make_components(),
            outputs,
            _make_inputs(),
        )

    def test_a_consumer_with_a_possessive_matches_its_party(self):
        outputs = _make_outputs()
        outputs[0] = outputs[0].model_copy(
            update={"consumer": "meeting attendees and my boss"}
        )
        _validate_diagram_structure(
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant AttendeesAndBoss as "Meeting Attendees and Boss: Receive"\n'
            "Order->>AttendeesAndBoss: start\n"
            "opt may happen\n"
            "Order->>AttendeesAndBoss: more\n"
            "end\n"
            "Order->>AttendeesAndBoss: outcome",
            _make_components(),
            outputs,
            _make_inputs(),
        )

    def test_a_participant_outside_the_record_names_the_known_ones(self):
        with self.assertRaises(ValidationError) as ctx:
            self._check(
                "sequenceDiagram\n"
                'participant Ghost as "Ghost: Haunt"\n'
                'participant Order as "Order: Process"\n'
                "Ghost->>Order: start\n"
                "Order-->>Ghost: ack\n"
                "opt may happen\n"
                "Ghost->>Order: more\n"
                "end\n"
                "Ghost->>Order: outcome"
            )

        self.assertIn("Ghost", str(ctx.exception))
        self.assertIn("not a component", str(ctx.exception))


class TestCreateDiagram(TestCase):
    """`_create_diagram` writes blobs to a temp directory.

    The directory it creates is not removed, so these tests run on a fake
    filesystem: a unit test must not leave anything in /tmp.
    """

    @override
    def setUp(self) -> None:
        self.setUpPyfakefs()

    @patch("workflows.evaluation_criteria.refine.refine")
    @patch.object(WorkflowSpec, "_generate_diagram")
    def test_materializes_blobs(self, mock_generate, mock_refine):
        workflow = WorkflowSpec.model_construct(
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
        result = WorkflowSpec._create_diagram(
            analysis=_make_analysis(),
            components=_make_components(),
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            session=session,
        )

        self.assertIsInstance(result, WorkflowSpec)
        mock_generate.assert_called_once()
        mock_refine.assert_called_once()

    @patch("workflows.evaluation_criteria.refine.refine")
    @patch.object(WorkflowSpec, "_generate_diagram")
    def test_refine_loop_reruns_on_changes(self, mock_generate, mock_refine):
        original = WorkflowSpec.model_construct(
            name="Test Workflow",
            diagram="sequenceDiagram\nA->>B: original",
            inputs=_make_inputs(),
            outputs=_make_outputs(),
            components=_make_components(),
            gap_analysis=None,
            architectural_validation="OK",
        )
        changed = WorkflowSpec.model_construct(
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
        result = WorkflowSpec._create_diagram(
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
