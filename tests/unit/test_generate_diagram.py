"""Tests for WorkflowSpec._generate_diagram and the diagram contract."""

import unittest
from unittest.mock import patch

from chat_workflow.exceptions import ValidationError
from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.models import Deliverable, Resource
from workflows.workflow.workflow_spec import (
    WorkflowSpec,
    _repair_diagram_outcome,
    _validate_diagram_structure,
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


class TestDiagramRepair(unittest.TestCase):
    """The final arrow is mechanical; the framework fixes its shape."""

    def test_a_self_loop_outcome_is_repaired_to_the_consumer(self):
        diagram = (
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant Inventory as "Inventory: Receive"\n'
            "Order->>Order: start\n"
            "Inventory-->>Order: ack\n"
            "opt may happen\n"
            "Order->>Order: more\n"
            "end\n"
            "Order->>Order: done"
        )

        repaired = _repair_diagram_outcome(diagram, _make_outputs())

        self.assertIn("Order->>Inventory: Send the final Test", repaired)
        _validate_diagram_structure(
            repaired, _make_components(), _make_outputs(), _make_inputs()
        )

    @patch.object(WorkflowSpec, "collect_all_rules", return_value=[])
    def test_the_model_repairs_the_outcome_before_validation(self, mock_collect):
        diagram = (
            "sequenceDiagram\n"
            'participant Order as "Order: Process"\n'
            'participant Inventory as "Inventory: Receive"\n'
            "Order->>Order: start\n"
            "Inventory-->>Order: ack\n"
            "opt may happen\n"
            "Order->>Order: more\n"
            "end\n"
            "Order->>Order: done"
        )

        spec = WorkflowSpec.model_validate(
            {
                "name": "Test Workflow",
                "diagram": diagram,
                "inputs": [item.model_dump() for item in _make_inputs()],
                "outputs": [item.model_dump() for item in _make_outputs()],
                "components": [item.model_dump() for item in _make_components()],
                "architectural_validation": "One owner, one artifact.",
            }
        )

        self.assertIn("Order->>Inventory: Send the final Test", spec.diagram)




if __name__ == "__main__":
    unittest.main(verbosity=2)
