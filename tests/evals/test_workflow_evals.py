"""Eval tests: verify LLM can produce workflow models.

These tests call a real LLM API and verify the LLM can produce each
workflow model with valid data. They run with ``make evals``.
"""

import unittest
from contextlib import suppress

from tests.conftest import timeout


class TestProcessAnalysisEval(unittest.TestCase):
    """Eval tests for ProcessAnalysis model."""

    @timeout(30)
    def test_llm_produces_process_analysis(self):
        """LLM should produce a ProcessAnalysis with all fields."""
        from tests.evals.helpers import run_one_shot_eval
        from workflows.workflow.models import ProcessAnalysis

        result = run_one_shot_eval(
            response_model=ProcessAnalysis,
            system_prompt=(
                "You are a Business Process Analyst. Analyze this process.\n\n"
                "Return phases, activities, orchestrating_component, and participants.\n"
                "Use intent='success' to return the complete ProcessAnalysis."
            ),
            initial_message=(
                "Customer places an order, the system validates payment, "
                "inventory is checked, and shipping is arranged."
            ),
            user_turn="Please analyze this process and return the result.",
        )
        self.assertIsInstance(result, ProcessAnalysis)
        self.assertTrue(len(result.phases) >= 1)
        self.assertTrue(len(result.activities) >= 1)
        self.assertTrue(len(result.orchestrating_component) > 0)
        self.assertTrue(len(result.participants) >= 1)

    @timeout(120)
    def test_multi_turn_conversation_with_user_bot(self):
        """ProcessAnalysis should complete efficiently with a realistic user bot."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow.models import ProcessAnalysis

        user_persona = (
            "You are a busy professional who attends lots of meetings. You take sketchy "
            "notes in a hurry and need help creating a repeatable process to turn those "
            "notes into proper meeting minutes with action items.\n\n"
            "You are an expert on your own meetings \u2014 you know who attends, what gets "
            "discussed, what decisions get made. But you know nothing about 'workflow "
            "decomposition', 'process phases', or 'components'. You just describe what "
            "happens naturally.\n\n"
            "The analyst you're talking to is trying to help you design a workflow you "
            "can use going forward. They're NOT trying to document your current ad-hoc "
            "process \u2014 they want to help you create something better.\n\n"
            "Respond helpfully to their questions using your knowledge of how your "
            "meetings work. Be patient but don't repeat yourself. If asked about "
            "something you don't understand (like abstract workflow concepts), ask "
            "them to explain in simpler terms."
        )

        result = run_multi_turn_eval(
            model_method=ProcessAnalysis.generate_from_chat,
            method_kwargs=dict(
                process_description="Writing up my sketchy meeting notes into a proper set of minutes with actions",
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, ProcessAnalysis)
        self.assertGreaterEqual(len(result.phases), 1)
        self.assertGreaterEqual(len(result.activities), 1)
        self.assertGreater(len(result.orchestrating_component), 0)
        self.assertGreaterEqual(len(result.participants), 1)


class TestGapAnalysisEval(unittest.TestCase):
    """Eval tests for GapAnalysis model."""

    @timeout(30)
    def test_llm_produces_gap_analysis(self):
        """LLM should produce a GapAnalysis from component requirements."""
        from tests.evals.helpers import run_one_shot_eval
        from workflows.workflow.models import GapAnalysis

        result = run_one_shot_eval(
            response_model=GapAnalysis,
            system_prompt=(
                "You are analyzing gaps in a component architecture.\n\n"
                "Return missing_components, missing_playbooks, integration_gaps, "
                "organizational_gaps, and recommendations.\n"
                "Use intent='success' to return the complete GapAnalysis."
            ),
            initial_message=(
                "Components needed: Order, Payment, Inventory. "
                "Existing: Order only. "
                "Payment integration is unclear with Inventory."
            ),
            user_turn="Analyze gaps and return the analysis.",
        )
        self.assertIsInstance(result, GapAnalysis)
        self.assertTrue(
            len(result.missing_components) > 0 or len(result.recommendations) > 0
        )


class TestInputEval(unittest.TestCase):
    """Eval tests for Input model."""

    @timeout(30)
    def test_llm_produces_input(self):
        """LLM should produce an Input model with all fields."""
        from tests.evals.helpers import run_one_shot_eval
        from workflows.workflow.models import Input

        result = run_one_shot_eval(
            response_model=Input,
            system_prompt=(
                "You are analyzing workflow inputs.\n\n"
                "Return source, format, trigger_conditions, dependencies, "
                "and validation_criteria.\n"
                "Use intent='success' to return the complete Input."
            ),
            initial_message=(
                "The order management workflow receives order data "
                "from the customer portal as JSON."
            ),
            user_turn="Return the input analysis.",
        )
        self.assertIsInstance(result, Input)
        self.assertTrue(len(result.source) > 0)
        self.assertTrue(len(result.format) > 0)
        self.assertTrue(len(result.trigger_conditions) > 0)

    @timeout(120)
    def test_multi_turn_input_generation(self):
        """Input.generate_from_chat should complete efficiently with a user bot."""
        from tests.evals.helpers import make_meeting_analysis, run_multi_turn_eval
        from workflows.workflow.models import Input

        user_persona = (
            "You are a busy professional. You know what information you need "
            "to write up meeting minutes (notes, attendee list, action items from "
            "last time). But you know nothing about 'workflow input analysis'. "
            "Describe what you start with in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )

        result = run_multi_turn_eval(
            model_method=Input.generate_from_chat,
            method_kwargs=dict(
                analysis=make_meeting_analysis(),
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


class TestOutputEval(unittest.TestCase):
    """Eval tests for Output model."""

    @timeout(30)
    def test_llm_produces_output(self):
        """LLM should produce an Output model with all fields."""
        from tests.evals.helpers import run_one_shot_eval
        from workflows.workflow.models import Output

        result = run_one_shot_eval(
            response_model=Output,
            system_prompt=(
                "You are analyzing workflow outputs.\n\n"
                "Return consumer, format, success_criteria, integration_points, "
                "and storage_requirements.\n"
                "Use intent='success' to return the complete Output."
            ),
            initial_message=(
                "The order confirmation is sent to the customer via email as a PDF."
            ),
            user_turn="Return the output analysis.",
        )
        self.assertIsInstance(result, Output)
        self.assertTrue(len(result.consumer) > 0)
        self.assertTrue(len(result.format) > 0)

    @timeout(120)
    def test_multi_turn_output_generation(self):
        """Output.generate_from_chat should complete efficiently with a user bot."""
        from tests.evals.helpers import make_meeting_analysis, run_multi_turn_eval
        from workflows.workflow.models import Output

        user_persona = (
            "You are a busy professional. You know what comes out of your "
            "meeting process: minutes, action items, decisions log. But you "
            "know nothing about 'workflow output analysis'. Describe what "
            "you produce in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )

        result = run_multi_turn_eval(
            model_method=Output.generate_from_chat,
            method_kwargs=dict(
                analysis=make_meeting_analysis(),
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


class TestComponentRequirementEval(unittest.TestCase):
    """Eval tests for ComponentRequirement model."""

    @timeout(30)
    def test_llm_produces_component_requirement(self):
        """LLM should produce a ComponentRequirement."""
        from tests.evals.helpers import run_one_shot_eval
        from workflows.workflow.models import ComponentRequirement

        result = run_one_shot_eval(
            response_model=ComponentRequirement,
            system_prompt=(
                "You are identifying business components.\n\n"
                "Return a ComponentRequirement with name (noun-based), purpose, "
                "required_inputs, expected_outputs, and component_type "
                "(one of: value_stream, artifact_producing, planning_service).\n"
                "Use intent='success' to return the complete ComponentRequirement."
            ),
            initial_message=(
                "We need a component that manages customer invoices "
                "throughout their lifecycle."
            ),
            user_turn="Return the component requirement.",
        )
        self.assertIsInstance(result, ComponentRequirement)
        self.assertTrue(len(result.name) > 0)
        self.assertTrue(len(result.purpose) > 0)
        self.assertIn(
            result.component_type,
            ["value_stream", "artifact_producing", "planning_service"],
        )

    @timeout(120)
    def test_multi_turn_component_identification(self):
        """ComponentRequirement.identify_from_chat should complete efficiently."""
        from tests.evals.helpers import make_meeting_analysis, run_multi_turn_eval
        from workflows.workflow.models import (
            ComponentRequirement,
            Input,
            Output,
        )

        analysis = make_meeting_analysis()
        inputs = [
            Input(
                source="Note Taker", format="Free-text notes",
                trigger_conditions="Meeting ends",
                validation_criteria="Contains date and key topics",
            ),
        ]
        outputs = [
            Output(
                consumer="Attendees", format="Formatted document",
                success_criteria="Accurate and timely",
                integration_points="Email",
                storage_requirements="Shared drive",
            ),
        ]
        user_persona = (
            "You are a busy professional. You know your meeting workflow "
            "well but know nothing about 'component architecture'. Describe "
            "the pieces of your process in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )

        result = run_multi_turn_eval(
            model_method=ComponentRequirement.identify_from_chat,
            method_kwargs=dict(
                analysis=analysis, inputs=inputs, outputs=outputs,
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


class TestGeneratedComponentEval(unittest.TestCase):
    """Eval tests for GeneratedComponent model."""

    @timeout(30)
    def test_llm_produces_python_code(self):
        """LLM should produce Python code via GeneratedComponent."""
        from tests.evals.helpers import run_one_shot_eval
        from workflows.workflow.models import GeneratedComponent

        result = run_one_shot_eval(
            response_model=GeneratedComponent,
            system_prompt=(
                "You are a Python code generator.\n\n"
                "Generate a complete Python file for a business component.\n"
                "Rules:\n"
                "- Import from pydantic import BaseModel, Field\n"
                "- Use Field(..., description=...) on all fields\n"
                "- One class per file\n"
                "- Valid Python\n\n"
                "Output format: Return ONLY the Python code as a string "
                "in the 'code' field.\n"
                "Use intent='success' to return the GeneratedComponent."
            ),
            initial_message=(
                "Create a component named 'Order' that manages customer orders. "
                "It needs fields: customer_name (str), items (list), total (float)."
            ),
            user_turn="Generate the Python code now.",
        )
        self.assertIsInstance(result, GeneratedComponent)
        self.assertTrue(len(result.code) > 0)
        self.assertIn("class ", result.code)
        self.assertIn("BaseModel", result.code)
        with suppress(SyntaxError):
            compile(result.code, "<test>", "exec")

    @timeout(120)
    def test_multi_turn_component_design(self):
        """Component._design_component should complete efficiently with a user bot."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow.component import Component
        from workflows.workflow.models import ComponentRequirement, GeneratedComponent

        req = ComponentRequirement(
            name="MinutesDraft",
            purpose="Transform raw meeting notes into structured minutes with action items",
            required_inputs=["Meeting notes"],
            expected_outputs=["Approved minutes", "Action items"],
            component_type="artifact_producing",
        )
        user_persona = (
            "You are a busy professional who writes up meeting minutes. "
            "You know exactly what makes good minutes: every action item "
            "must have an owner and due date, decisions must be recorded, "
            "and minutes must be distributed within 24 hours. "
            "But you know NOTHING about Python, Pydantic, or programming. "
            "If the architect uses technical terms, ask them to explain "
            "in plain language."
            "\n\nRespond helpfully but don't repeat yourself."
        )

        result = run_multi_turn_eval(
            model_method=Component._design_component,
            method_kwargs=dict(
                requirements=req, max_turns=10,
            ),
            user_persona=user_persona,
            judge_prompt=(
                "Evaluate this conversation between a software architect and a "
                "business user designing a component. Did the architect:\n"
                "- Ask about what makes good output (quality criteria)?\n"
                "- Translate the user's domain knowledge into validation rules?\n"
                "- Avoid technical jargon or explain it when asked?\n"
                "- Synthesize and propose (proposing then asking to fill details "
                "is GOOD, not bad)?\n\n"
                "Answer YES for good conversation, NO if the architect was stuck "
                "in a pure questioning loop or failed to elicit quality criteria."
            ),
        )
        self.assertIsInstance(result, GeneratedComponent)
        self.assertGreater(len(result.code), 0)
        self.assertIn("class ", result.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
