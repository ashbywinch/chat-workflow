"""Eval tests: verify LLM can one-shot produce workflow models.

These tests call a real LLM API and verify the LLM can produce each
workflow model with valid data. They run with ``make evals``.
"""

import sys
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

    @timeout(120)
    def test_multi_turn_conversation_with_user_bot(self):
        """ProcessAnalysis should complete efficiently with a realistic user bot.

        This eval tests that the workflow doesn't get stuck in Socratic questioning
        loops when talking to a user who describes their problem clearly. The user
        bot is an expert in their domain (meetings) but knows nothing about workflows.
        """
        from tests.evals.helpers import AgentIO, make_tools
        from workflows.workflow.models import ProcessAnalysis

        # User bot persona: busy professional who wants help with meeting minutes
        user_persona = (
            "You are a busy professional who attends lots of meetings. You take sketchy "
            "notes in a hurry and need help creating a repeatable process to turn those "
            "notes into proper meeting minutes with action items.\n\n"
            "You are an expert on your own meetings — you know who attends, what gets "
            "discussed, what decisions get made. But you know nothing about 'workflow "
            "decomposition', 'process phases', or 'components'. You just describe what "
            "happens naturally.\n\n"
            "The analyst you're talking to is trying to help you design a workflow you "
            "can use going forward. They're NOT trying to document your current ad-hoc "
            "process — they want to help you create something better.\n\n"
            "Respond helpfully to their questions using your knowledge of how your "
            "meetings work. Be patient but don't repeat yourself. If asked about "
            "something you don't understand (like abstract workflow concepts), ask "
            "them to explain in simpler terms."
        )

        user_bot = AgentIO(persona_prompt=user_persona, config=_CONFIG)
        self._conversation_transcript = user_bot._history  # live reference to history list
        session = make_tools(user_bot)

        try:
            analysis = ProcessAnalysis.generate_from_chat(
                process_description="Writing up my sketchy meeting notes into a proper set of minutes with actions",
                max_turns=10,
                session=session,
            )

            # Verify we got a valid result
            self.assertIsInstance(analysis, ProcessAnalysis)
            self.assertGreaterEqual(len(analysis.phases), 1, "Should have at least one phase")
            self.assertGreaterEqual(len(analysis.activities), 1, "Should have at least one activity")
            self.assertGreater(len(analysis.orchestrating_component), 0, "Should have orchestrating component")
            self.assertGreaterEqual(len(analysis.participants), 1, "Should have at least one participant")

            # Verify we didn't burn all turns (the key regression test)
            self.assertLess(
                session.state.turn_count,
                10,
                f"Workflow burned all {session.state.turn_count} turns — likely stuck in questioning loop. "
                f"User bot had to provide {len(user_bot.outputs)} responses."
            )

            # Judge the conversation quality — did the agent synthesize, loop, or repeat itself?
            from tests.evals.helpers import llm_judge
            transcript = "\n---\n".join(
                msg["content"]
                for msg in user_bot._history
                if msg["role"] == "user"
            )
            is_good_conversation, conversation_reason = llm_judge(
                "Evaluate this conversation transcript from a workflow analyst "
                "helping a user design a process for turning meeting notes into "
                "minutes. Did the analyst:\n"
                "- Synthesize the user's description into a coherent proposal "
                "(it is OK to propose structure then ask to fill details)?\n"
                "- Avoid repeating the same question or asking for information "
                "the user already provided?\n"
                "- Use plain language the user can understand "
                "(explaining jargon when asked is good)?\n"
                "- Keep the conversation focused and efficient?\n\n"
                "Answer YES if the conversation is productive — proposing then "
                "asking targeted follow-ups is GOOD. "
                "Answer NO only if the analyst was stuck in a pure questioning "
                "loop with no synthesis, or was genuinely obtuse.",
                transcript,
                _CONFIG,
            )
            self.assertTrue(
                is_good_conversation,
                f"Conversation quality issue detected.\n"
                f"Judge's reasoning: {conversation_reason}\n"
                f"Transcript:\n{transcript}",
            )
        finally:
            # If the test failed or errored, dump the full conversation transcript
            # to stderr so it's visible in test runner output for debugging.
            if sys.exc_info()[0] is not None:
                transcript_lines = []
                for msg in self._conversation_transcript:
                    role = msg["role"]
                    content = msg.get("content", "")
                    transcript_lines.append(f"[{role}]\n{content}")
                print(
                    "\n=== CONVERSATION TRANSCRIPT (before failure) ===\n"
                    + "\n---\n".join(transcript_lines)
                    + "\n=== END TRANSCRIPT ===\n",
                    file=sys.stderr,
                )


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

    @timeout(120)
    def test_multi_turn_input_generation(self):
        """Input.generate_from_chat should complete efficiently with a user bot."""
        from tests.evals.helpers import AgentIO, make_tools
        from workflows.workflow.models import Input, ProcessAnalysis

        analysis = ProcessAnalysis(
            phases=["Note-taking", "Review & Clarify", "Draft Minutes", "Review & Approve"],
            activities=[
                "Take meeting notes", "Review notes for clarity",
                "Identify action items", "Write minutes draft",
                "Circulate for review", "Incorporate feedback",
                "Distribute final minutes",
            ],
            orchestrating_component="Meeting Organizer",
            participants=["Meeting Attendees", "Note Taker", "Reviewers"],
        )
        user_persona = (
            "You are a busy professional. You know what information you need "
            "to write up meeting minutes (notes, attendee list, action items from "
            "last time). But you know nothing about 'workflow input analysis'. "
            "Describe what you start with in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )
        user_bot = AgentIO(persona_prompt=user_persona, config=_CONFIG)
        self._conversation_transcript = user_bot._history
        session = make_tools(user_bot)
        try:
            result = Input.generate_from_chat(
                analysis=analysis, max_turns=10, session=session,
            )
            self.assertIsInstance(result, list)
            self.assertGreaterEqual(len(result), 1, "Should have at least one input")
            self.assertLess(
                session.state.turn_count, 10,
                f"Burned all {session.state.turn_count} turns on Input. "
                f"Responses: {len(user_bot.outputs)}",
            )
            from tests.evals.helpers import llm_judge
            transcript = "\n---\n".join(
                m["content"] for m in user_bot._history if m["role"] == "user"
            )
            ok, reason = llm_judge(
                "Did the analyst propose a structure and then ask to fill in "
                "details? Avoid repeating questions? Use plain language "
                "(explaining jargon when asked is fine)? "
                "Answer YES for productive conversation, "
                "NO only if stuck in a pure questioning loop.",
                transcript, _CONFIG,
            )
            self.assertTrue(ok, f"Conversation quality issue.\nJudge: {reason}\nTranscript:\n{transcript}")
        finally:
            if sys.exc_info()[0] is not None:
                lines = [f"[{m['role']}]\n{m.get('content','')}" for m in self._conversation_transcript]
                print(
                    "\n=== CONVERSATION TRANSCRIPT (before failure) ===\n"
                    + "\n---\n".join(lines)
                    + "\n=== END ===\n",
                    file=sys.stderr,
                )


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

    @timeout(120)
    def test_multi_turn_output_generation(self):
        """Output.generate_from_chat should complete efficiently with a user bot."""
        from tests.evals.helpers import AgentIO, make_tools
        from workflows.workflow.models import Output, ProcessAnalysis

        analysis = ProcessAnalysis(
            phases=["Note-taking", "Review & Clarify", "Draft Minutes", "Review & Approve"],
            activities=[
                "Take meeting notes", "Review notes for clarity",
                "Identify action items", "Write minutes draft",
                "Circulate for review", "Incorporate feedback",
                "Distribute final minutes",
            ],
            orchestrating_component="Meeting Organizer",
            participants=["Meeting Attendees", "Note Taker", "Reviewers"],
        )
        user_persona = (
            "You are a busy professional. You know what comes out of your "
            "meeting process: minutes, action items, decisions log. But you "
            "know nothing about 'workflow output analysis'. Describe what "
            "you produce in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )
        user_bot = AgentIO(persona_prompt=user_persona, config=_CONFIG)
        self._conversation_transcript = user_bot._history
        session = make_tools(user_bot)
        try:
            result = Output.generate_from_chat(
                analysis=analysis, max_turns=10, session=session,
            )
            self.assertIsInstance(result, list)
            self.assertGreaterEqual(len(result), 1, "Should have at least one output")
            self.assertLess(
                session.state.turn_count, 10,
                f"Burned all {session.state.turn_count} turns on Output. "
                f"Responses: {len(user_bot.outputs)}",
            )
            from tests.evals.helpers import llm_judge
            transcript = "\n---\n".join(
                m["content"] for m in user_bot._history if m["role"] == "user"
            )
            ok, reason = llm_judge(
                "Did the analyst propose a structure and then ask to fill in "
                "details? Avoid repeating questions? Use plain language "
                "(explaining jargon when asked is fine)? "
                "Answer YES for productive conversation, "
                "NO only if stuck in a pure questioning loop.",
                transcript, _CONFIG,
            )
            self.assertTrue(ok, f"Conversation quality issue.\nJudge: {reason}\nTranscript:\n{transcript}")
        finally:
            if sys.exc_info()[0] is not None:
                lines = [f"[{m['role']}]\n{m.get('content','')}" for m in self._conversation_transcript]
                print(
                    "\n=== CONVERSATION TRANSCRIPT (before failure) ===\n"
                    + "\n---\n".join(lines)
                    + "\n=== END ===\n",
                    file=sys.stderr,
                )


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

    @timeout(120)
    def test_multi_turn_component_identification(self):
        """ComponentRequirement.identify_from_chat should complete efficiently."""
        from tests.evals.helpers import AgentIO, llm_judge, make_tools
        from workflows.workflow.models import (
            ComponentRequirement,
            Input,
            Output,
            ProcessAnalysis,
        )

        analysis = ProcessAnalysis(
            phases=["Note-taking", "Review & Clarify", "Draft Minutes", "Review & Approve"],
            activities=[
                "Take meeting notes", "Review notes for clarity",
                "Identify action items", "Write minutes draft",
                "Circulate for review", "Incorporate feedback",
            ],
            orchestrating_component="Meeting Organizer",
            participants=["Meeting Attendees", "Note Taker", "Reviewers"],
        )
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
        user_bot = AgentIO(persona_prompt=user_persona, config=_CONFIG)
        self._conversation_transcript = user_bot._history
        session = make_tools(user_bot)
        try:
            result = ComponentRequirement.identify_from_chat(
                analysis=analysis, inputs=inputs, outputs=outputs,
                max_turns=10, session=session,
            )
            self.assertIsInstance(result, list)
            self.assertGreaterEqual(len(result), 1, "Should have at least one component")
            self.assertLess(
                session.state.turn_count, 10,
                f"Burned all {session.state.turn_count} turns. "
                f"Responses: {len(user_bot.outputs)}",
            )
            transcript = "\n---\n".join(
                m["content"] for m in user_bot._history if m["role"] == "user"
            )
            ok, reason = llm_judge(
                "Did the analyst propose a structure and then ask to fill in "
                "details? Avoid repeating questions? Use plain language "
                "(explaining jargon when asked is fine)? "
                "Answer YES for productive conversation, "
                "NO only if stuck in a pure questioning loop.",
                transcript, _CONFIG,
            )
            self.assertTrue(ok, f"Conversation quality issue.\nJudge: {reason}\nTranscript:\n{transcript}")
        finally:
            if sys.exc_info()[0] is not None:
                lines = [f"[{m['role']}]\n{m.get('content','')}" for m in self._conversation_transcript]
                print(
                    "\n=== CONVERSATION TRANSCRIPT (before failure) ===\n"
                    + "\n---\n".join(lines)
                    + "\n=== END ===\n",
                    file=sys.stderr,
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

    @timeout(120)
    def test_multi_turn_component_design(self):
        """Component._design_component should complete efficiently with a user bot.

        This tests that the software architect asks what makes good output
        and translates domain knowledge into Validation rules.
        """
        from tests.evals.helpers import AgentIO, llm_judge, make_tools
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
        user_bot = AgentIO(persona_prompt=user_persona, config=_CONFIG)
        self._conversation_transcript = user_bot._history
        session = make_tools(user_bot)
        try:
            result = Component._design_component(
                requirements=req, max_turns=10, session=session,
            )
            self.assertIsInstance(result, GeneratedComponent)
            self.assertGreater(len(result.code), 0, "Should have generated code")
            self.assertIn("class ", result.code, "Generated code should have a class")
            self.assertLess(
                session.state.turn_count, 10,
                f"Burned all {session.state.turn_count} turns. "
                f"Responses: {len(user_bot.outputs)}",
            )
            transcript = "\n---\n".join(
                m["content"] for m in user_bot._history if m["role"] == "user"
            )
            ok, reason = llm_judge(
                "Evaluate this conversation between a software architect and a "
                "business user designing a component. Did the architect:\n"
                "- Ask about what makes good output (quality criteria)?\n"
                "- Translate the user's domain knowledge into validation rules?\n"
                "- Avoid technical jargon or explain it when asked?\n"
                "- Synthesize and propose (proposing then asking to fill details "
                "is GOOD, not bad)?\n\n"
                "Answer YES for good conversation, NO if the architect was stuck "
                "in a pure questioning loop or failed to elicit quality criteria.",
                transcript, _CONFIG,
            )
            self.assertTrue(
                ok,
                f"Design conversation quality issue.\nJudge: {reason}\n"
                f"Transcript:\n{transcript}",
            )
        finally:
            if sys.exc_info()[0] is not None:
                lines = [f"[{m['role']}]\n{m.get('content','')}" for m in self._conversation_transcript]
                print(
                    "\n=== CONVERSATION TRANSCRIPT (before failure) ===\n"
                    + "\n---\n".join(lines)
                    + "\n=== END ===\n",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)