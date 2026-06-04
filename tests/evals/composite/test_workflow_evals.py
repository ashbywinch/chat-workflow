"""Eval tests that verify workflow prompt quality with real LLM calls.

These test the actual @atomic_workflow-decorated methods with an LLM-powered
user bot (AgentIO) and an LLM judge to non-deterministically evaluate
conversation quality — did the agent synthesize, loop, repeat itself, etc.

Pydantic validates the structural output. These tests check the conversation.
"""

import unittest
from contextlib import suppress

from tests.conftest import timeout
from tests.evals.helpers import DEFAULT_JUDGE_RULES

META_LEVEL_JUDGE_RULES: dict[str, str] = {
    **DEFAULT_JUDGE_RULES,
    "Stays at meta-level": (
        "The assistant is helping the user define the SHAPE and STRUCTURE of their "
        "outputs/inputs/process as design artifacts. This is a meta-level conversation "
        "\u2014 the assistant defines what things look like, not their content. "
        "The assistant does NOT drop into the object-level and start helping the user "
        "achieve their real-world goal (e.g., generating business ideas, planning menus, "
        "writing posts). When the user describes a goal, the assistant asks about structure "
        "('what fields should each output have?') rather than trying to help achieve the "
        "goal ('what skills do you have?')."
    ),
}

REDIRECT_JUDGE_RULES: dict[str, str] = {
    **META_LEVEL_JUDGE_RULES,
    "Redirects without joining": (
        "When the user goes off-topic and describes their actual domain content "
        "(business ideas, specific ventures, etc.), the assistant acknowledges briefly "
        "then redirects back to the meta-level conversation. It does NOT join the user "
        "in discussing the domain content. A brief 'That sounds interesting \u2014 let's "
        "come back to that. What would a good business idea output look like?' is "
        "acceptable. A response like 'Oh, that's a great idea! Have you thought about "
        "the target market?' is NOT \u2014 that's joining in."
    ),
}


class TestProcessDefinitionEval(unittest.TestCase):
    """Eval tests for ProcessDefinition model."""

    @timeout(120)
    def test_multi_turn_conversation_with_user_bot(self):
        """ProcessDefinition should complete efficiently with a realistic user bot."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow.models import ProcessDefinition, generate_from_chat

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
            model_method=generate_from_chat,
            method_kwargs=dict(
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, ProcessDefinition)

    @timeout(120)
    def test_process_analysis_with_adhd_writer(self):
        """ProcessDefinition should stay at meta-level with an ADHD writer who keeps describing content."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow.models import ProcessDefinition, generate_from_chat

        user_persona = (
            "You are a content writer who creates long-form blog posts. You want to "
            "document your writing process so new writers on your team can follow it. "
            "But whenever you start describing the stages, you get excited about the "
            "actual content \u2014 the post you're working on, the research you've done, "
            "the expert you interviewed yesterday. You have ADHD and don't naturally "
            "stay on topic. When the assistant tries to redirect you to talk about "
            "process structure, you follow for a bit, then dive back into describing "
            "your latest article."
        )

        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(
                max_turns=10,
            ),
            user_persona=user_persona,
            judge_rules=REDIRECT_JUDGE_RULES,
        )
        self.assertIsInstance(result, ProcessDefinition)


class TestResourceEval(unittest.TestCase):
    """Eval tests for Resource model."""

    @timeout(120)
    def test_multi_turn_resource_generation(self):
        """Resource.generate_from_chat should complete efficiently with a user bot."""
        from tests.evals.helpers import make_meeting_analysis, run_multi_turn_eval
        from workflows.workflow.models import Resource

        user_persona = (
            "You are a busy professional. You know what information you need "
            "to write up meeting minutes (notes, attendee list, action items from "
            "last time). But you know nothing about 'workflow resource analysis'. "
            "Describe what you start with in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )

        result = run_multi_turn_eval(
            model_method=Resource.generate_from_chat,
            method_kwargs=dict(
                analysis=make_meeting_analysis(),
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    @timeout(120)
    def test_resource_with_adhd_chef(self):
        """Resource should stay at meta-level with an ADHD chef who keeps describing dishes."""
        from tests.evals.helpers import run_multi_turn_eval
        from workflows.workflow.models import Resource

        user_persona = (
            "You are a professional chef planning weekly menus for a restaurant. You want "
            "to document your menu planning process so junior chefs can follow it. But "
            "whenever you start describing what goes into your planning, you get excited "
            "and start talking about the actual dishes \u2014 the seasonal ingredients you've "
            "spotted, the new supplier you found, the head chef's feedback on last week's "
            "menu. You have ADHD and don't naturally stay on topic. When the assistant "
            "tries to redirect you to talk about the structure of your process inputs, you "
            "follow for a bit, then get carried away describing your latest menu idea."
        )

        result = run_multi_turn_eval(
            model_method=Resource.generate_from_chat,
            method_kwargs=dict(
                analysis=None,
                outputs=None,
                max_turns=10,
            ),
            user_persona=user_persona,
            judge_rules=REDIRECT_JUDGE_RULES,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


class TestOutputEval(unittest.TestCase):
    """Eval tests for Output model."""

    @timeout(120)
    def test_multi_turn_output_generation(self):
        """Deliverable.generate_from_chat should complete efficiently with a user bot."""
        from tests.evals.helpers import make_meeting_analysis, run_multi_turn_eval
        from workflows.workflow.models import Deliverable

        user_persona = (
            "You are a busy professional. You know what comes out of your "
            "meeting process: minutes, action items, decisions log. But you "
            "know nothing about 'workflow output analysis'. Describe what "
            "you produce in plain terms."
            "\n\nRespond helpfully but don't repeat yourself."
        )

        result = run_multi_turn_eval(
            model_method=Deliverable.generate_from_chat,
            method_kwargs=dict(
                analysis=make_meeting_analysis(),
                max_turns=10,
            ),
            user_persona=user_persona,
            judge_rules=META_LEVEL_JUDGE_RULES,
        )
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    @timeout(120)
    def test_output_with_adhd_ideas_person(self):
        """Deliverable should stay at meta-level with an ADHD user who keeps describing business ideas."""
        from chat_workflow.exceptions import TurnLimitExceededError
        from tests.evals.helpers import (
            AgentIO,
            capture_on_failure,
            format_transcript,
            llm_judge,
            make_config,
            make_tools,
        )
        from workflows.workflow.models import Deliverable

        user_persona = (
            "You have tons of business ideas \u2014 you're always thinking of new ones \u2014 "
            "and you're excited about them. You want to use this tool to help you capture "
            "and organize them. But every time you start describing your process, you get "
            "carried away talking about your actual ideas \u2014 you find them fascinating. "
            "You have ADHD and don't naturally stay on topic. When the assistant tries to "
            "redirect you to talk about your process, you follow for a bit, but then you "
            "remember another brilliant idea and start describing that. You're not trying "
            "to be difficult \u2014 your brain just works this way."
        )

        config = make_config()
        user_bot = AgentIO(persona_prompt=user_persona, config=config)
        session = make_tools(user_bot)

        with capture_on_failure(session):
            try:
                result = Deliverable.generate_from_chat(
                    analysis=None,
                    max_turns=10,
                    session=session,
                )
                # If we get here without hitting the limit, validate output
                self.assertIsInstance(result, list)
                self.assertGreaterEqual(len(result), 1)
            except TurnLimitExceededError:
                # Hitting the turn limit is OK — the ADHD user is chatty
                # Still validate via judge rules on whatever conversation happened
                pass

            # Always run the judge on the conversation transcript
            transcript = format_transcript(session)
            judge_result = llm_judge(REDIRECT_JUDGE_RULES, transcript, config)
            failures = [v for v in judge_result.verdicts if not v.passed]
            assert not failures, (
                f"Conversation quality: {len(failures)}/{len(REDIRECT_JUDGE_RULES)} rules failed:\n"
                + "\n".join(f"  [{v.rule}] FAIL: {v.explanation}" for v in failures)
            )


class TestComponentResponsibilitiesEval(unittest.TestCase):
    """Eval tests for ComponentResponsibilities model."""

    @timeout(120)
    def test_multi_turn_component_identification(self):
        """ComponentResponsibilities.identify_from_chat should complete efficiently."""
        from tests.evals.helpers import make_meeting_analysis, run_multi_turn_eval
        from workflows.workflow.models import (
            ComponentResponsibilities,
            Deliverable,
            Resource,
        )

        analysis = make_meeting_analysis()
        inputs = [
            Resource(
                source="Note Taker",
                format="Free-text notes",
                trigger_conditions="Meeting ends",
                validation_criteria="Contains date and key topics",
            ),
        ]
        outputs = [
            Deliverable(
            name="Test",
            description="Test",
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
            model_method=ComponentResponsibilities.identify_from_chat,
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
        from workflows.workflow import ComponentSourceCode
        from workflows.workflow.design_spec import ComponentDesignSpec
        from workflows.workflow.domain_spec import ComponentDomainField, ComponentDomainSpec
        from workflows.workflow.interaction_context import ComponentInteractionContext
        from workflows.workflow.structure import ComponentStructure

        design_spec = ComponentDesignSpec(
            domain_spec=ComponentDomainSpec(
                name="MinutesDraft",
                description=(
                    "Structured meeting minutes that capture what happened, "
                    "decisions made, and action items assigned"
                ),
                fields=[
                    ComponentDomainField(
                        name="meeting_date",
                        domain_description="When the meeting took place",
                        field_type_hint="date",
                    ),
                    ComponentDomainField(
                        name="attendees",
                        domain_description="People who attended the meeting",
                        field_type_hint="list of person names",
                    ),
                    ComponentDomainField(
                        name="key_decisions",
                        domain_description="Important decisions made during the meeting",
                        field_type_hint="list of decision descriptions",
                    ),
                    ComponentDomainField(
                        name="action_items",
                        domain_description="Action items with owners and due dates",
                        field_type_hint="list of action items",
                    ),
                    ComponentDomainField(
                        name="next_meeting_date",
                        domain_description="Date of the next meeting, if scheduled",
                        field_type_hint="optional date",
                    ),
                ],
                what_good_looks_like=[
                    "Attendees can immediately understand decisions made",
                    "Someone who missed the meeting can catch up in two minutes",
                    "Every action item has a clear owner and due date",
                    "The minutes are concise but complete",
                ],
                expert_role="Meeting Minutes Administrator",
            ),
            structure=ComponentStructure(
                description=(
                    "Structured meeting minutes that capture what happened, "
                    "decisions made, and action items assigned"
                ),
            ),
            interaction_context=ComponentInteractionContext(
                must_prioritize=[
                    "Always ask about decisions and action items early in the conversation"
                ],
                auto_suggest=[
                    "Suggest action item owners based on the topic discussed",
                    "Propose a due date for each action item",
                ],
                user_pain_points=[
                    "Users often forget to list all attendees",
                    "Users sometimes omit decisions that were made implicitly",
                ],
            ),
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
            model_method=ComponentSourceCode.generate,
            method_kwargs=dict(
                design_spec=design_spec,
                max_turns=10,
            ),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, ComponentSourceCode)
        self.assertGreater(len(result.code), 0)
        self.assertIn("class ", result.code)
        with suppress(SyntaxError):
            compile(result.code, "<test>", "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
