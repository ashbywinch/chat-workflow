"""Eval test that verifies the full Workflow.create() pipeline end-to-end.

This test calls the real @composite_workflow Workflow.create() with an LLM-powered
user bot, then verifies:
- The result is a valid Workflow
- All components are ComponentResponsibilities (not ComponentRequirement)
- Each component has scope_description and incidental_notes fields
- The conversation quality passes an LLM judge
"""

import unittest

from tests.conftest import timeout


class TestWorkflowPipelineEval(unittest.TestCase):
    """Eval tests for the full Workflow.create() pipeline."""

    @timeout(300)
    def test_full_workflow_pipeline(self):
        """Workflow.create() should complete end-to-end with ComponentResponsibilities."""
        from tests.evals.helpers import (
            AgentIO,
            capture_on_failure,
            format_transcript,
            llm_judge,
            make_config,
            make_tools,
        )
        from workflows.workflow import Workflow
        from workflows.workflow.component_responsibilities import (
            ComponentResponsibilities,
        )

        config = make_config()
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
            "them to explain in simpler terms.\n\n"
            "IMPORTANT: When asked about incidental notes for a component, you can "
            "either provide relevant internal details or say 'None' if you have nothing "
            "to add. Be brief."
        )

        user_bot = AgentIO(persona_prompt=user_persona, config=config)
        session = make_tools(user_bot)

        judge_rules = {
            "Completes workflow": (
                "The agent successfully completed the full workflow creation process "
                "including process analysis, input/output identification, component "
                "identification, gap resolution, and diagram generation."
            ),
            "Captures incidental notes": (
                "The agent asked the user about incidental notes for each identified "
                "component and captured their response."
            ),
            "No repetition": (
                "The agent didn't ask for the same information again after the user already provided it."
            ),
            "Uses expertise": (
                "The agent made informed proposals based on what the user said, rather "
                "than asking the user to describe everything from scratch."
            ),
        }

        with capture_on_failure(session, label="workflow_pipeline"):
            result = Workflow.create(
                process_description=("Writing up my sketchy meeting notes into a proper set of minutes with actions"),
                session=session,
                max_refinements=1,
            )

            # Verify result type
            self.assertIsInstance(result, Workflow)
            self.assertGreater(len(result.components), 0)

            # Verify all components are ComponentResponsibilities
            for comp in result.components:
                self.assertIsInstance(comp, ComponentResponsibilities)
                self.assertIsInstance(comp.scope_description, str)
                self.assertGreater(len(comp.scope_description), 0)
                # incidental_notes is a string (may be empty)
                self.assertIsInstance(comp.incidental_notes, str)

            # LLM judge on the full conversation
            transcript = format_transcript(session)
            judge_result = llm_judge(judge_rules, transcript, config)
            failures = [v for v in judge_result.verdicts if not v.passed]
            assert not failures, (
                f"Conversation quality: {len(failures)}/{len(judge_rules)} "
                f"rules failed:\n" + "\n".join(f"  [{v.rule}] FAIL: {v.explanation}" for v in failures)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
