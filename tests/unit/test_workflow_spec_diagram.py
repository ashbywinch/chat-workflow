"""The diagram contract and the mechanical repairs.

The model writes a Mermaid sequence diagram from the record. The contract is
decided here, not by a judge: participants, hand-offs, a decision, a return
flow, and the final outcome to the consumer.
"""

import unittest

from chat_workflow.exceptions import ValidationError
from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.models.deliverable import Deliverable
from workflows.workflow.models.resource import Resource
from workflows.workflow.workflow_spec import (
    _fallback_diagram,
    _matches_known,
    _name_words,
    _normalise_name,
    _primary_output,
    _repair_diagram_outcome,
    _repair_generic_participants,
    _validate_diagram_structure,
)

COMPONENTS = [
    ComponentSpec(
        name="Meeting Minutes",
        purpose="Write the minutes.",
        scope_description="Not the agenda.",
        owner_role="You",
        required_inputs=["Raw notes"],
        component_type="artifact_producing",
        incidental_notes="",
    )
]
OUTPUTS = [
    Deliverable(
        name="Meeting Minutes",
        description="The record of the meeting.",
        consumer="Meeting Attendees",
        format="A document",
        success_criteria="Every decision has an owner.",
        integration_points="They receive it by email.",
        storage_requirements="Shared drive",
    )
]
INPUTS = [
    Resource(
        source="Raw notes",
        format="Bullet points",
        dependencies=[],
        validation_criteria="Every decision is captured.",
    )
]

VALID = """sequenceDiagram
    participant Minutes as "Meeting Minutes: Produce"
    participant Attendees as "Meeting Attendees: Receive"
    Minutes->>Minutes: Write up the decisions
    Minutes->>Attendees: Send the draft
    Attendees->>Minutes: Return corrections
    opt Review needed
        Minutes->>Attendees: Send the draft
    end
    Minutes->>Attendees: Send the final Meeting Minutes
"""


class TestNames(unittest.TestCase):
    def test_names_compare_without_noise(self):
        self.assertEqual(_normalise_name("  Meeting_Minutes! "), "meeting minutes")
        self.assertEqual(_name_words("The Meeting Attendees"), ["meeting", "attendees"])

    def test_a_shortened_party_matches(self):
        known = {"meeting attendees", "raw notes"}
        self.assertTrue(_matches_known("Attendees", known))
        self.assertTrue(_matches_known("Meeting Attendees", known))
        self.assertFalse(_matches_known("Finance", known))


class TestDiagramContract(unittest.TestCase):
    def test_a_valid_diagram_passes(self):
        _validate_diagram_structure(VALID, COMPONENTS, OUTPUTS, INPUTS)

    def _rejects(self, diagram: str, message: str) -> None:
        with self.assertRaises(ValidationError) as ctx:
            _validate_diagram_structure(diagram, COMPONENTS, OUTPUTS, INPUTS)
        self.assertIn(message, str(ctx.exception))

    def test_no_participants(self):
        self._rejects("sequenceDiagram\n", "declares no participants")

    def test_an_undeclared_participant(self):
        self._rejects(
            VALID + "    Minutes->>Ghost: Hello\n", "not declared"
        )

    def test_no_messages(self):
        self._rejects(
            'sequenceDiagram\n    participant Minutes as "Meeting Minutes: Produce"\n',
            "has no messages",
        )

    def test_only_self_loops(self):
        self._rejects(
            'sequenceDiagram\n    participant Minutes as "Meeting Minutes: Produce"\n'
            "    Minutes->>Minutes: Work\n",
            "self-loop",
        )

    def test_no_decision_point(self):
        self._rejects(
            'sequenceDiagram\n    participant Minutes as "Meeting Minutes: Produce"\n'
            '    participant Attendees as "Meeting Attendees: Receive"\n'
            "    Minutes->>Attendees: Send the final Meeting Minutes\n",
            "no decision point",
        )

    # lucidlint: ignore fakefs str.replace edits a string here, not os.replace — the rule mistakes the one for the other
    def test_a_long_message(self):
        self._rejects(
            VALID.replace(
                "Send the draft",
                "Send the draft " + "x" * 70,
            ),
            "is long",
        )

    # lucidlint: ignore fakefs str.replace edits a string here, not os.replace — the rule mistakes the one for the other
    def test_an_unknown_participant_path(self):
        diagram = VALID.replace('"Meeting Attendees: Receive"', '"Finance: Receive"')
        diagram = diagram.replace("Attendees", "Finance")
        self._rejects(diagram, "is not a component")


class TestRepairs(unittest.TestCase):
    def test_the_primary_output_is_the_external_consumer(self):
        outputs = [
            {"name": "Private Notes", "consumer": "You"},
            {"name": "Meeting Minutes", "consumer": "Meeting Attendees"},
        ]
        self.assertEqual(
            _primary_output(outputs, COMPONENTS), ("Meeting Minutes", "Meeting Attendees")
        )

    def test_no_output_has_no_primary(self):
        self.assertIsNone(_primary_output([], COMPONENTS))

    def test_a_generic_participant_becomes_the_component(self):
        diagram = (
            'sequenceDiagram\n    participant You as "You: Note-taker"\n'
            "    You->>You: Work\n"
        )
        repaired = _repair_generic_participants(diagram, COMPONENTS, OUTPUTS)

        self.assertIn('"Meeting Minutes: Note-taker"', repaired)
        self.assertNotIn('"You:', repaired)

    def test_a_component_label_is_left_alone(self):
        repaired = _repair_generic_participants(VALID, COMPONENTS, OUTPUTS)
        self.assertEqual(repaired.strip(), VALID.strip())

    def test_the_outcome_is_appended_after_a_return_flow(self):
        diagram = (
            'sequenceDiagram\n    participant Minutes as "Meeting Minutes: Produce"\n'
            '    participant Attendees as "Meeting Attendees: Receive"\n'
            "    Minutes->>Attendees: Send the draft\n"
            "    Attendees->>Minutes: Return corrections\n"
        )
        repaired = _repair_diagram_outcome(diagram, OUTPUTS, COMPONENTS)

        self.assertTrue(repaired.rstrip().endswith("Send the final Meeting Minutes"))

    def test_the_outcome_is_not_duplicated(self):
        repaired = _repair_diagram_outcome(VALID, OUTPUTS, COMPONENTS)
        self.assertEqual(repaired.count("Send the final Meeting Minutes"), 1)

    def test_a_fallback_diagram_is_valid(self):
        diagram = _fallback_diagram(COMPONENTS, OUTPUTS)

        self.assertIn("sequenceDiagram", diagram)
        _validate_diagram_structure(diagram, COMPONENTS, OUTPUTS, INPUTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
