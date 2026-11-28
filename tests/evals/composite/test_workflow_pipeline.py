"""Eval test that verifies the full WorkflowSpec.create() pipeline end-to-end.

This test calls the real @composite_workflow_step WorkflowSpec.create() with an LLM-powered
user bot, then verifies:
- The result is a valid Workflow
- All components are ComponentSpec (not ComponentSpec)
- Each component has scope_description and incidental_notes fields
- The conversation quality passes an LLM judge
"""

# lucidlint: ignore-file fakefs the checkpoint under .sisyphus/ is read by the resume eval in another process

import json
import tempfile
import unittest
from pathlib import Path

from tests.conftest import timeout


class TestWorkflowPipelineEval(unittest.TestCase):
    """Eval tests for the full WorkflowSpec.create() pipeline."""

    @timeout(7200)
    def test_full_workflow_pipeline(self):
        """WorkflowSpec.create() should complete end-to-end with ComponentSpec."""
        from tests.evals.helpers import (
            AgentIO,
            capture_on_failure,
            format_transcript,
            llm_judge,
            make_config,
            make_tools,
        )
        from tests.evals.personas import MEETING_MINUTES_PERSONA
        from workflows.workflow import WorkflowSpec
        from workflows.workflow.component_spec import (
            ComponentSpec,
        )

        config = make_config()
        user_persona = MEETING_MINUTES_PERSONA

        user_bot = AgentIO(persona_prompt=user_persona, config=config)
        session = make_tools(user_bot)

        judge_rules = {
            "Completes workflow": (
                "The agent successfully completed the full workflow creation process "
                "including process analysis, input/output identification, component "
                "identification, gap resolution, and diagram generation."
            ),
            "Captures incidental notes": (
                "If the user mentions incidental details about a part — a "
                "constraint, a person, a recurring problem — the result keeps "
                "them. The agent is not required to ask for them: the field "
                "holds what the user mentioned."
            ),
            "No repetition": (
                "The agent didn't ask for the same information again after the user already provided it."
            ),
            "Uses expertise": (
                "The agent made informed proposals based on what the user said, rather "
                "than asking the user to describe everything from scratch."
            ),
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            capture_on_failure(session, label="workflow_pipeline"),
        ):
            result = WorkflowSpec.create(
                process_description=(
                    "Writing up my sketchy meeting notes into a proper set of minutes with actions"
                ),
                session=session,
                max_refinements=1,
                output_dir=Path(tmpdir),
            )

            # Verify result type
            self.assertIsInstance(result, WorkflowSpec)
            self.assertGreater(len(result.components), 0)

            # Checkpoint the component specs so a component failure can be
            # resumed with tests.evals.composite.test_resume_component_generation
            # instead of re-running the whole pipeline.
            checkpoint_dir = Path(".sisyphus/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "workflow_pipeline-latest.json").write_text(
                json.dumps([comp.model_dump() for comp in result.components], indent=2) + "\n"
            )

            # Every component must actually be generated. WorkflowSpec.create
            # logs failures and continues; the test must not pass on a partial
            # build.
            failed = [line for line in user_bot.outputs if "✗ Failed to create" in line]
            assert not failed, "Component generation failed:\n" + "\n".join(failed)

            # Verify all components are ComponentSpec
            for comp in result.components:
                self.assertIsInstance(comp, ComponentSpec)
                self.assertIsInstance(comp.scope_description, str)
                self.assertGreater(len(comp.scope_description), 0)
                # incidental_notes is a string (may be empty)
                self.assertIsInstance(comp.incidental_notes, str)

            # LLM judge on the conversation the user actually had. Framework
            # corrections and validation feedback are harness text, and feeding
            # them to the judge biases it towards agreeing with them.
            transcript = format_transcript(session, audience="user")
            judge_result = llm_judge(judge_rules, transcript, config)
            failures = [v for v in judge_result.verdicts if not v.passed]
            assert not failures, (
                f"Conversation quality: {len(failures)}/{len(judge_rules)} "
                f"rules failed:\n" + "\n".join(f"  [{v.rule}] FAIL: {v.explanation}" for v in failures)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
