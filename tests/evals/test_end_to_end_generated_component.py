"""End-to-end eval: ComponentResponsibilities → Component.create() → generated component → import → run → judge.

This test:
1. Creates a realistic ComponentResponsibilities for MinutesDraft
2. Runs Component.create() — all 4 phases execute (explore → design → gather → generate)
3. Reads the generated code from disk, verifies syntax
4. Dynamically imports the generated module
5. Runs the imported component's workflow with AgentIO + LLM judge
6. Verifies the output Pydantic model validates
"""

import importlib
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path

from tests.conftest import timeout
from tests.evals.helpers import (
    AgentIO,
    capture_on_failure,
    make_config,
    make_tools,
    run_multi_turn_eval,
)
from workflows.workflow.component import Component
from workflows.workflow.component_responsibilities import ComponentResponsibilities


def make_minutes_draft_responsibilities() -> ComponentResponsibilities:
    """Create a realistic ComponentResponsibilities for a MinutesDraft artifact."""
    return ComponentResponsibilities(
        name="MinutesDraft",
        purpose="Creates structured meeting minutes",
        scope_description=(
            "Represents the formal written record of a meeting, including what was discussed, "
            "decisions made, and action items assigned. Does NOT represent the meeting agenda, "
            "pre-meeting materials, or follow-up communications."
        ),
        required_inputs=["meeting_notes", "attendee_list"],
        component_type="artifact_producing",
        incidental_notes=(
            "The user typically takes sketchy notes during meetings and needs help "
            "turning them into proper structured minutes. They want the assistant to "
            "ask clarifying questions and fill in gaps."
        ),
    )


class TestEndToEndGeneratedComponentEval(unittest.TestCase):
    """End-to-end eval: full pipeline from responsibilities to running generated component."""

    @timeout(300)
    def test_end_to_end_generated_component(self):
        """Full pipeline: ComponentResponsibilities → Component.create() → import → run → judge."""
        config = make_config()
        responsibilities = make_minutes_draft_responsibilities()

        # --- Phase A: Run Component.create() with all 4 phases ---
        # The user persona must work across all design phases (explore, design, gather).
        # The user is a meeting coordinator who knows their domain but nothing about code.
        design_user_persona = (
            "You are a busy professional who takes sketchy notes during meetings and needs "
            "help turning them into proper structured minutes. You know your meetings well "
            "but know nothing about data structures, programming, or technical concepts.\n\n"
            "The meeting you're thinking of is a weekly team sync. Key details:\n"
            "- Attendees: Sarah (PM), Mike (Design), Lisa (Engineering), and yourself\n"
            "- You discuss project status, blockers, and next steps\n"
            "- Decisions are made about priorities and deadlines\n"
            "- Action items are assigned with owners\n\n"
            "When the assistant asks about what information the minutes should capture, "
            "describe what you need in natural business terms. For example: 'I need to "
            "record who attended, what we decided, and who's doing what by when.'\n\n"
            "When the assistant asks about rules or constraints, think about what makes "
            "good minutes vs bad minutes. For example: 'Every decision should say why we "
            "made it, not just what we decided.'\n\n"
            "When the assistant asks about how they should help you, think about what "
            "you'd want an assistant to do. For example: 'Remind me if I forget to "
            "assign an owner to an action item.'\n\n"
            "Respond naturally and helpfully. Be patient but don't repeat yourself. "
            "If asked about something technical, say you don't understand and ask them "
            "to explain in simpler terms."
        )

        design_user_bot = AgentIO(persona_prompt=design_user_persona, config=config)
        design_session = make_tools(design_user_bot)

        with tempfile.TemporaryDirectory() as tmpdir:
            with capture_on_failure(design_session, label="component_create"):
                component = Component.create(
                    requirements=responsibilities,
                    session=design_session,
                    output_dir=Path(tmpdir),
                )

            # Verify the Component record
            self.assertIsInstance(component, Component)
            self.assertEqual(component.name, "MinutesDraft")
            self.assertTrue(component.code_path.exists())

            # Read generated code
            generated_code = component.code_path.read_text()
            self.assertGreater(len(generated_code), 0)
            self.assertIn("class ", generated_code)

            # Verify syntax compiles
            compile(generated_code, "<generated>", "exec")

            # --- Phase B: Dynamically import the generated module ---
            sys.path.insert(0, tmpdir)

            try:
                module_name = "minutesdraft"
                spec = importlib.util.spec_from_file_location(
                    module_name, str(component.code_path)
                )
                self.assertIsNotNone(
                    spec, f"Could not create spec for {component.code_path}"
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)

                # Find the workflow class and method
                from pydantic import BaseModel

                workflow_method = None
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if not isinstance(attr, type):
                        continue
                    if not (isinstance(attr, type) and issubclass(attr, BaseModel)):
                        continue
                    if attr is BaseModel:
                        continue
                    for method_name in dir(attr):
                        method = getattr(attr, method_name, None)
                        if method is None:
                            continue
                        if getattr(method, "_is_workflow", False):
                            workflow_method = method
                            break
                    if workflow_method is not None:
                        break

                classes_found = [
                    n
                    for n in dir(mod)
                    if isinstance(getattr(mod, n), type)
                    and issubclass(getattr(mod, n), BaseModel)
                    and getattr(mod, n) is not BaseModel
                ]
                self.assertIsNotNone(
                    workflow_method,
                    f"No @atomic_workflow method found. Classes: {classes_found}",
                )

                # --- Phase C: Run the generated workflow with a user bot ---
                user_persona = (
                    "You are a busy professional who just finished a project kickoff meeting. "
                    "You took sketchy notes and need help turning them into proper minutes.\n\n"
                    "The meeting was about the Q3 marketing campaign. Key points:\n"
                    "- Attended by: Sarah (PM), Mike (Design), Lisa (Engineering)\n"
                    "- Decided to launch the campaign on August 1st\n"
                    "- Budget is $50,000\n"
                    "- Action items: Sarah will draft the creative brief by Friday, "
                    "Mike will create mockups by next Wednesday, "
                    "Lisa will estimate engineering hours by Thursday\n"
                    "- Next meeting scheduled for next Monday\n\n"
                    "You know your meeting well but know nothing about data structures "
                    "or programming. Respond naturally to the assistant's questions. "
                    "Be patient but don't repeat yourself."
                )

                judge_rules = {
                    "Completes the conversation": (
                        "The agent successfully completed the conversation and produced "
                        "a structured result. It did not get stuck in a loop or fail "
                        "to produce output."
                    ),
                    "No repetition": (
                        "The agent didn't get stuck asking the same question more than "
                        "twice after the user already provided the information."
                    ),
                    "Uses domain language": (
                        "The agent used natural business language (meeting, decisions, "
                        "action items) rather than technical terms (fields, records, schemas)."
                    ),
                }

                result = run_multi_turn_eval(
                    model_method=workflow_method,
                    method_kwargs=dict(max_turns=8),
                    user_persona=user_persona,
                    judge_rules=judge_rules,
                    config=config,
                )

                # --- Step 6: Verify the result is a valid Pydantic model ---
                self.assertIsNotNone(result)
                self.assertIsInstance(result, BaseModel)

                # Verify it has content
                result_dict = result.model_dump() if hasattr(result, "model_dump") else {}
                self.assertGreater(len(str(result_dict)), 0)

            finally:
                # Clean up sys.path and sys.modules
                with suppress(ValueError):
                    sys.path.remove(tmpdir)
                with suppress(KeyError):
                    del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main(verbosity=2)
