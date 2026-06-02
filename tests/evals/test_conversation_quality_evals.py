"""Eval tests for conversation quality: confusion handling and jargon avoidance."""

from __future__ import annotations

import unittest

from tests.conftest import timeout
from tests.evals.helpers import (
    run_multi_turn_eval,
)
from workflows.workflow.component_responsibilities import ComponentRequirement
from workflows.workflow.models import (
    Deliverable,
    ProcessDefinition,
    Resource,
)
from workflows.workflow.models.gap_analysis import GapAnalysis

# ── Personas ──

CONFUSED_USER_PERSONA = (
    "You are someone who needs help planning meals for your unusual dietary requirements. "
    "You are an expert on your own needs — you know exactly what you do and don't eat, "
    "what kitchen equipment you have, and what you're able to cook given your disabilities. "
    "But you have never had an office job and you have a relatively low level of education.\n\n"
    "You are openly confused by business or technical jargon. When the assistant uses words "
    "like 'consumer', 'format', 'success criteria', 'inputs', 'outputs', 'phases', or "
    "'activities', you don't just ignore it — you ask for clarification: 'what does that "
    "mean?' or 'I don't understand what you're asking.' You will keep pushing back until "
    "the assistant explains in plain language.\n\n"
    "You are not trying to be difficult — you genuinely don't know what these terms mean "
    "and you're not the type to pretend you understand and guess. You expect the assistant "
    "to talk to you like a real person helping you out, not like a consultant analyzing "
    "your 'process.'"
)

# ── Judge Rules ──

CONFUSION_JUDGE_RULES = {
    "No business jargon": (
        "FAIL if the assistant used any of these EXACT words: 'output', "
        "'deliverable', 'process', 'workflow', 'consumer', "
        "'success criteria', 'integration', 'storage', "
        "'input', 'resource', 'action item', 'stakeholder', "
        "'identified'. "
        "PASS otherwise."
    ),
}

JARGON_FREE_RULES = {
    "No field name leakage": (
        "PASS if the assistant used plain English equivalents like 'who "
        "uses this' (instead of 'consumer'), 'what it looks like' (instead "
        "of 'format'), 'how you know it is good' (instead of 'success "
        "criteria'). These equivalents are GOOD and should be praised.\n"
        "FAIL only if the assistant used the EXACT field name (consumer, "
        "format, success_criteria, integration_points, "
        "storage_requirements, source, trigger_conditions, dependencies, "
        "validation_criteria, phases, activities, orchestrating_component, "
        "participants) as a standalone word. "
        "Short common words like 'form', 'source', 'phase' used in "
        "regular English are always fine. "
        "PASS otherwise."
    ),
}


# ── Test: Confusion Handling ──


class TestConfusionHandlingEval(unittest.TestCase):
    """User bot that is confused by jargon — agent must simplify language."""

    @timeout(120)
    def test_output_generation_with_confused_user(self):
        """Deliverable.generate_from_chat drops jargon when user is confused."""
        result = run_multi_turn_eval(
            model_method=Deliverable.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=CONFUSION_JUDGE_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)

    @timeout(120)
    def test_resource_generation_with_confused_user(self):
        """Resource.generate_from_chat drops jargon when user is confused."""
        result = run_multi_turn_eval(
            model_method=Resource.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=CONFUSION_JUDGE_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)


# ── Test: Jargon-Free ──


class TestJargonFreeEval(unittest.TestCase):
    """Conversation transcripts must not contain model field names."""

    @timeout(120)
    def test_output_jargon_free(self):
        """Deliverable conversation uses plain language, not field names."""
        result = run_multi_turn_eval(
            model_method=Deliverable.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)

    @timeout(120)
    def test_resource_jargon_free(self):
        """Resource conversation uses plain language, not field names."""
        result = run_multi_turn_eval(
            model_method=Resource.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)

    @timeout(120)
    def test_process_definition_jargon_free(self):
        """ProcessDefinition conversation uses plain language, not field names."""
        from workflows.workflow.models import generate_from_chat

        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs={
                "max_turns": 8,
            },
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, ProcessDefinition)

    @timeout(120)
    def test_component_requirement_jargon_free(self):
        """ComponentRequirement conversation uses plain language, not field names."""
        analysis = ProcessDefinition(
            phases=["Planning", "Execution"],
            activities=["Plan meals", "Shop", "Cook", "Review"],
            orchestrating_component="Meal Planner",
            participants=["Cook", "Family Members"],
        )
        inputs = [
            Resource(
                source="Dietary preferences",
                format="List",
                trigger_conditions="Weekly planning session",
                dependencies=[],
                validation_criteria="All restrictions captured",
            ),
        ]
        outputs = [
            Deliverable(
                consumer="Family",
                format="Weekly meal plan",
                success_criteria="Diet-compliant, balanced",
                integration_points="Kitchen prep",
                storage_requirements="Kitchen wall / fridge",
            ),
        ]
        result = run_multi_turn_eval(
            model_method=ComponentRequirement.identify_from_chat,
            method_kwargs={
                "analysis": analysis,
                "inputs": inputs,
                "outputs": outputs,
                "max_turns": 8,
            },
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        self.assertIsInstance(result, list)

    @timeout(120)
    def test_gap_analysis_jargon_free(self):
        """GapAnalysis conversation uses plain language, not field names."""
        analysis = ProcessDefinition(
            phases=["Planning", "Execution"],
            activities=["Plan meals", "Shop", "Cook", "Review"],
            orchestrating_component="Meal Planner",
            participants=["Cook", "Family Members"],
        )
        requirements = [
            ComponentRequirement(
                name="MealPlan",
                purpose="Plan weekly meals",
                required_inputs=["Dietary preferences"],
                expected_outputs=["Weekly meal plan"],
                component_type="artifact_producing",
            ),
        ]
        result = run_multi_turn_eval(
            model_method=GapAnalysis.analyze_from_chat,
            method_kwargs={
                "components": requirements,
                "analysis": analysis,
                "max_turns": 8,
            },
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, GapAnalysis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
