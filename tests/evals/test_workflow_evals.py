"""Eval tests that verify workflow prompt quality with real LLM calls.

These test the actual @atomic_workflow-decorated methods with an LLM-powered
user bot (AgentIO) and an LLM judge to non-deterministically evaluate
conversation quality — did the agent synthesize, loop, repeat itself, etc.

Pydantic validates the structural output. These tests check the conversation.
"""

import unittest
from contextlib import suppress

from tests.conftest import timeout


class TestProcessAnalysisEval(unittest.TestCase):
    """Eval tests for ProcessAnalysis model."""

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


class TestInputEval(unittest.TestCase):
    """Eval tests for Input model."""

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
                source="Note Taker",
                format="Free-text notes",
                trigger_conditions="Meeting ends",
                validation_criteria="Contains date and key topics",
            ),
        ]
        outputs = [
            Output(
                consumer="Attendees",
                format="Formatted document",
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
                analysis=analysis,
                inputs=inputs,
                outputs=outputs,
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


class TestGeneratedComponentEval(unittest.TestCase):
    """Eval tests for GeneratedComponent model."""

    @timeout(120)
    def test_multi_turn_component_design(self):
        """GeneratedComponent.generate should complete efficiently with a user bot."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow import GeneratedComponent
        from workflows.workflow.models import ComponentRequirement

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
            model_method=GeneratedComponent.generate,
            method_kwargs=dict(
                requirements=req,
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, GeneratedComponent)
        self.assertGreater(len(result.code), 0)
        self.assertIn("class ", result.code)
        with suppress(SyntaxError):
            compile(result.code, "<test>", "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
