"""Eval tests: verify LLM can one-shot produce workflow models.

These tests call a real LLM API and verify the LLM can produce each
workflow model with valid data. They run with ``make evals``.
"""

import unittest
from contextlib import suppress
from pathlib import Path

from chat_workflow import (
    AgentResponse,
    AtomicWorkflow,
    AtomicWorkflowConfig,
    AtomicWorkflowFailedError,
    Config,
    TurnResult,
)
from tests.conftest import timeout

_CONFIG = Config(Path(__file__).parent.parent.parent / "config.json")


class TestProcessAnalysisEval(unittest.TestCase):
    """Eval tests for ProcessAnalysis model."""

    @timeout(30)
    def test_llm_produces_process_analysis(self):
        """LLM should produce a ProcessAnalysis with all fields."""
        from workflows.workflow.models import ProcessAnalysis

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are a Business Process Analyst. Analyze this process.\n\n"
                    "Return phases, activities, orchestrating_component, and participants.\n"
                    "Use intent='success' to return the complete ProcessAnalysis."
                ),
                response_model=AgentResponse[ProcessAnalysis],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": (
                            "Customer places an order, the system validates payment, "
                            "inventory is checked, and shipping is arranged."
                        ),
                    }
                ],
                on_continue=lambda action: TurnResult[ProcessAnalysis].continuing(action.message or ""),
                on_success=lambda action: TurnResult[ProcessAnalysis].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
            )
        )

        result = orchestrator.process_turn("Please analyze this process and return the result.")

        self.assertIsNotNone(result)
        if result.result:
            analysis = result.result
            self.assertIsInstance(analysis, ProcessAnalysis)
            self.assertTrue(len(analysis.phases) >= 1)
            self.assertTrue(len(analysis.activities) >= 1)
            self.assertTrue(len(analysis.orchestrating_component) > 0)
            self.assertTrue(len(analysis.participants) >= 1)


class TestGapAnalysisEval(unittest.TestCase):
    """Eval tests for GapAnalysis model."""

    @timeout(30)
    def test_llm_produces_gap_analysis(self):
        """LLM should produce a GapAnalysis from component requirements."""
        from workflows.workflow.models import GapAnalysis

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are analyzing gaps in a component architecture.\n\n"
                    "Return missing_components, missing_playbooks, integration_gaps, "
                    "organizational_gaps, and recommendations.\n"
                    "Use intent='success' to return the complete GapAnalysis."
                ),
                response_model=AgentResponse[GapAnalysis],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": (
                            "Components needed: Order, Payment, Inventory. "
                            "Existing: Order only. "
                            "Payment integration is unclear with Inventory."
                        ),
                    }
                ],
                on_continue=lambda action: TurnResult[GapAnalysis].continuing(action.message or ""),
                on_success=lambda action: TurnResult[GapAnalysis].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
            )
        )

        result = orchestrator.process_turn("Analyze gaps and return the analysis.")

        self.assertIsNotNone(result)
        if result.result:
            gaps = result.result
            self.assertIsInstance(gaps, GapAnalysis)
            # Should have at least some gaps identified
            self.assertTrue(
                len(gaps.missing_components) > 0 or len(gaps.recommendations) > 0
            )


class TestInputEval(unittest.TestCase):
    """Eval tests for Input model."""

    @timeout(30)
    def test_llm_produces_input(self):
        """LLM should produce an Input model with all fields."""
        from workflows.workflow.models import Input

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are analyzing workflow inputs.\n\n"
                    "Return source, format, trigger_conditions, dependencies, "
                    "and validation_criteria.\n"
                    "Use intent='success' to return the complete Input."
                ),
                response_model=AgentResponse[Input],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": (
                            "The order management workflow receives order data "
                            "from the customer portal as JSON."
                        ),
                    }
                ],
                on_continue=lambda action: TurnResult[Input].continuing(action.message or ""),
                on_success=lambda action: TurnResult[Input].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
            )
        )

        result = orchestrator.process_turn("Return the input analysis.")

        self.assertIsNotNone(result)
        if result.result:
            inp = result.result
            self.assertIsInstance(inp, Input)
            self.assertTrue(len(inp.source) > 0)
            self.assertTrue(len(inp.format) > 0)
            self.assertTrue(len(inp.trigger_conditions) > 0)


class TestOutputEval(unittest.TestCase):
    """Eval tests for Output model."""

    @timeout(30)
    def test_llm_produces_output(self):
        """LLM should produce an Output model with all fields."""
        from workflows.workflow.models import Output

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are analyzing workflow outputs.\n\n"
                    "Return consumer, format, success_criteria, integration_points, "
                    "and storage_requirements.\n"
                    "Use intent='success' to return the complete Output."
                ),
                response_model=AgentResponse[Output],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": (
                            "The order confirmation is sent to the customer "
                            "via email as a PDF."
                        ),
                    }
                ],
                on_continue=lambda action: TurnResult[Output].continuing(action.message or ""),
                on_success=lambda action: TurnResult[Output].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
            )
        )

        result = orchestrator.process_turn("Return the output analysis.")

        self.assertIsNotNone(result)
        if result.result:
            out = result.result
            self.assertIsInstance(out, Output)
            self.assertTrue(len(out.consumer) > 0)
            self.assertTrue(len(out.format) > 0)


class TestComponentRequirementEval(unittest.TestCase):
    """Eval tests for ComponentRequirement model."""

    @timeout(30)
    def test_llm_produces_component_requirement(self):
        """LLM should produce a ComponentRequirement."""
        from workflows.workflow.models import ComponentRequirement

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
                system_prompt=(
                    "You are identifying business components.\n\n"
                    "Return a ComponentRequirement with name (noun-based), purpose, "
                    "required_inputs, expected_outputs, and component_type "
                    "(one of: value_stream, artifact_producing, planning_service).\n"
                    "Use intent='success' to return the complete ComponentRequirement."
                ),
                response_model=AgentResponse[ComponentRequirement],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": (
                            "We need a component that manages customer invoices "
                            "throughout their lifecycle."
                        ),
                    }
                ],
                on_continue=lambda action: TurnResult[ComponentRequirement].continuing(action.message or ""),
                on_success=lambda action: TurnResult[ComponentRequirement].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
            )
        )

        result = orchestrator.process_turn("Return the component requirement.")

        self.assertIsNotNone(result)
        if result.result:
            req = result.result
            self.assertIsInstance(req, ComponentRequirement)
            self.assertTrue(len(req.name) > 0)
            self.assertTrue(len(req.purpose) > 0)
            self.assertIn(
                req.component_type,
                ["value_stream", "artifact_producing", "planning_service"],
            )


class TestGeneratedComponentEval(unittest.TestCase):
    """Eval tests for GeneratedComponent model."""

    @timeout(30)
    def test_llm_produces_python_code(self):
        """LLM should produce Python code via GeneratedComponent."""
        from workflows.workflow.models import GeneratedComponent

        orchestrator = AtomicWorkflow(
            config=AtomicWorkflowConfig(
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
                response_model=AgentResponse[GeneratedComponent],
                max_turns=3,
                model=_CONFIG.model,
                provider=_CONFIG.provider,
                max_retries=_CONFIG.max_retries,
                request_timeout_seconds=_CONFIG.request_timeout_seconds,
                initial_messages=[
                    {
                        "role": "user",
                        "content": (
                            "Create a component named 'Order' that manages customer orders. "
                            "It needs fields: customer_name (str), items (list), total (float)."
                        ),
                    }
                ],
                on_continue=lambda action: TurnResult[GeneratedComponent].continuing(action.message or ""),
                on_success=lambda action: TurnResult[GeneratedComponent].success(action.result),
                on_failure=lambda action: AtomicWorkflowFailedError(action.message or "No reason given"),
            )
        )

        result = orchestrator.process_turn("Generate the Python code now.")

        self.assertIsNotNone(result)
        if result.result:
            gen = result.result
            self.assertIsInstance(gen, GeneratedComponent)
            self.assertTrue(len(gen.code) > 0)
            self.assertIn("class ", gen.code)
            self.assertIn("BaseModel", gen.code)

            with suppress(SyntaxError):
                compile(gen.code, "<test>", "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)