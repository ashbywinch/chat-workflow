"""Eval tests for conversation quality: confusion handling and jargon avoidance."""

from __future__ import annotations

import unittest

from tests.conftest import timeout
from tests.evals.helpers import (
    run_multi_turn_eval,
)
from workflows.workflow.component_responsibilities import ComponentRequirement
from workflows.workflow.models import (
    Input,
    Output,
    ProcessAnalysis,
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
    "Simplifies on confusion": (
        "When the user expresses confusion or says 'I don't understand', "
        "the assistant simplifies their language rather than repeating the same "
        "question or re-explaining with different words. The assistant should "
        "drop technical framing and use plain conversational language."
    ),
    "No jargon escalation": (
        "If the user doesn't understand a term like 'outputs', the assistant should "
        "not introduce other technical terms like 'deliverables' or 'artifacts'. "
        "They should just say 'what you want to end up with' or similar plain language."
    ),
}

JARGON_FREE_RULES = {
    "No field name leakage": (
        "The assistant did not use any of these model field names when talking "
        "to the user: consumer, format, success_criteria, integration_points, "
        "storage_requirements, source, trigger_conditions, validation_criteria, "
        "dependencies, phases, activities, orchestrating_component, participants. "
        "It's acceptable to use plain language equivalents like 'who uses this' "
        "instead of 'consumer', 'what it looks like' instead of 'format', etc."
    ),
}


# ── Test: Confusion Handling ──


class TestConfusionHandlingEval(unittest.TestCase):
    """User bot that is confused by jargon — agent must simplify language."""

    @timeout(120)
    def test_output_generation_with_confused_user(self):
        """Output.generate_from_chat drops jargon when user is confused."""
        result = run_multi_turn_eval(
            model_method=Output.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=CONFUSION_JUDGE_RULES,
        )
        self.assertIsInstance(result, list)

    @timeout(120)
    def test_input_generation_with_confused_user(self):
        """Input.generate_from_chat drops jargon when user is confused."""
        result = run_multi_turn_eval(
            model_method=Input.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=CONFUSION_JUDGE_RULES,
        )
        self.assertIsInstance(result, list)


# ── Test: Jargon-Free ──


class TestJargonFreeEval(unittest.TestCase):
    """Conversation transcripts must not contain model field names."""

    @timeout(120)
    def test_output_jargon_free(self):
        """Output conversation uses plain language, not field names."""
        result = run_multi_turn_eval(
            model_method=Output.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        self.assertIsInstance(result, list)

    @timeout(120)
    def test_input_jargon_free(self):
        """Input conversation uses plain language, not field names."""
        result = run_multi_turn_eval(
            model_method=Input.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        self.assertIsInstance(result, list)

    @timeout(120)
    def test_process_analysis_jargon_free(self):
        """ProcessAnalysis conversation uses plain language, not field names."""
        result = run_multi_turn_eval(
            model_method=ProcessAnalysis.generate_from_chat,
            method_kwargs={
                "process_description": "Planning weekly meals for a family with dietary restrictions",
                "max_turns": 8,
            },
            user_persona=CONFUSED_USER_PERSONA,
            judge_rules=JARGON_FREE_RULES,
        )
        self.assertIsInstance(result, ProcessAnalysis)

    @timeout(120)
    def test_component_requirement_jargon_free(self):
        """ComponentRequirement conversation uses plain language, not field names."""
        analysis = ProcessAnalysis(
            phases=["Planning", "Execution"],
            activities=["Plan meals", "Shop", "Cook", "Review"],
            orchestrating_component="Meal Planner",
            participants=["Cook", "Family Members"],
        )
        inputs = [
            Input(
                source="Dietary preferences",
                format="List",
                trigger_conditions="Weekly planning session",
                dependencies=[],
                validation_criteria="All restrictions captured",
            ),
        ]
        outputs = [
            Output(
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
        analysis = ProcessAnalysis(
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
        self.assertIsInstance(result, GapAnalysis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
