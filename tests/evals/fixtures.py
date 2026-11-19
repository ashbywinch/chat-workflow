"""Eval fixtures: the input objects each step is handed.

These are inputs, not model output, but they are built from the same models
the judge validates, so an invalid fixture breaks an eval before it starts.
They therefore live in one module with one checker:

    scripts/check_eval_fixtures.py

Run it after changing a rule or a fixture. Fixing a fixture is the right
response to a failure here — the input should be valid, and passing invalid
input to a step tests the wrong thing.
"""

from __future__ import annotations

from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.models import Deliverable, ProcessDefinition, Resource


def meeting_minutes_deliverable() -> Deliverable:
    """What a note-taker produces: one written record of a meeting."""
    return Deliverable(
        name="Meeting Minutes",
        description=(
            "The written record of the meeting: decisions made, action items "
            "assigned, and key discussion points."
        ),
        consumer="Meeting Attendees",
        format="Formatted document (PDF or shared doc)",
        success_criteria=(
            "Every decision is recorded and every action item names an owner."
        ),
        integration_points="Emailed to attendees, who review and follow up.",
        storage_requirements="Shared drive, one folder per meeting series",
    )


def meeting_notes_resource() -> Resource:
    return Resource(
        source="Note Taker",
        format="Free-text notes",
        validation_criteria=(
            "Contains the meeting date, the attendee list, and at least one "
            "discussion topic."
        ),
    )


def family_dietary_preferences() -> Resource:
    return Resource(
        source="Family members",
        format="List of dietary preferences",
        dependencies=[],
        validation_criteria=(
            "Lists every dietary restriction for each family member by name."
        ),
    )


def weekly_meal_plan_deliverable() -> Deliverable:
    """What a family cook produces: a plan for the week's meals."""
    return Deliverable(
        name="Weekly Meal Plan",
        description=(
            "Seven days of meals for the family, with each day's dinner and "
            "the groceries it needs."
        ),
        consumer="Family",
        format="One-page plan",
        success_criteria=(
            "Each day of the week lists a dinner, and every ingredient in it "
            "is on the family's allowed list."
        ),
        integration_points=(
            "The family receives the plan and reads it before shopping and "
            "cooking."
        ),
        storage_requirements="Kitchen wall / fridge",
    )


def weekly_menu_deliverable() -> Deliverable:
    """What a restaurant kitchen produces: a menu for one week."""
    return Deliverable(
        name="Weekly Menu",
        description=None,
        consumer="The kitchen team and diners",
        format="A printed menu and a kitchen prep sheet",
        success_criteria=(
            "Before the menu is printed, every day of the week has an entry, "
            "every dish lists its ingredients, and every dish names the station "
            "that prepares it."
        ),
        integration_points=(
            "The kitchen team receives it to prepare the dishes; front of house "
            "receives a printed copy to answer guest questions."
        ),
        storage_requirements="Kept in the kitchen binder and on the shared drive.",
    )


def meal_planning_analysis() -> ProcessDefinition:
    return ProcessDefinition(
        phases=["Planning", "Shopping", "Cooking"],
        activities=["Plan meals", "Shop", "Cook", "Review"],
        orchestrating_component="Home Cook",
        participants=["Cook", "Family Members"],
    )


def meeting_analysis() -> ProcessDefinition:
    return ProcessDefinition(
        phases=["Note-taking", "Review & Clarify", "Draft Minutes"],
        activities=[
            "Take meeting notes",
            "Review notes for clarity",
            "Identify action items",
            "Write minutes draft",
        ],
        orchestrating_component="Meeting Minutes Coordinator",
        participants=["Meeting Attendees", "Note Taker", "Reviewers"],
    )


def weekly_meal_plan_spec() -> ComponentSpec:
    return ComponentSpec(
        name="Weekly Meal Plan",
        purpose=(
            "Produces the weekly meal plan — which meals are cooked on which "
            "days and the groceries they require — for the family."
        ),
        required_inputs=["Dietary preferences"],
        scope_description=(
            "Creates and updates the weekly meal plan (including the associated "
            "grocery list) for one week. Does not include shopping, cooking, or "
            "reviewing the plan."
        ),
        component_type="artifact_producing",
        owner_role="Home Cook",
        incidental_notes="",
    )


def minutes_draft_spec() -> ComponentSpec:
    """The meeting-minutes component a note-taker owns."""
    return ComponentSpec(
        name="MinutesDraft",
        purpose=(
            "Produces the formal written record of a meeting — decisions made "
            "and action items assigned — for the meeting attendees."
        ),
        scope_description=(
            "Produces and maintains the formal written record of a meeting: it "
            "turns the meeting notes and attendee list into a record of what was "
            "discussed, the decisions made, and the action items assigned. It "
            "does NOT produce the meeting agenda, pre-meeting materials, or "
            "follow-up communications."
        ),
        required_inputs=["meeting_notes", "attendee_list"],
        component_type="artifact_producing",
        owner_role="Note Taker",
        incidental_notes=(
            "The user typically takes sketchy notes during meetings and needs "
            "help turning them into proper structured minutes. They want the "
            "assistant to ask clarifying questions and fill in gaps."
        ),
    )


def meeting_minutes_spec() -> ComponentSpec:
    return ComponentSpec(
        name="MeetingMinutes",
        purpose=(
            "Produces meeting minutes that capture decisions made, action items "
            "assigned, and key discussion points for meeting attendees."
        ),
        scope_description=(
            "Captures what happened in a meeting: decisions made, action items "
            "assigned, key discussion points. Does NOT include pre-meeting "
            "preparation or post-meeting follow-up tracking."
        ),
        required_inputs=["Raw meeting notes", "Attendee list"],
        component_type="artifact_producing",
        owner_role="Note Taker",
        incidental_notes="",
    )
