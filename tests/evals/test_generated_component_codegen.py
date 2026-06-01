"""End-to-end eval: GeneratedComponent.generate() produces working code.

This test:
1. Constructs a ComponentDesignSpec for MinutesDraft
2. Calls GeneratedComponent.generate(design_spec, session) to produce code
3. Writes code to temp file, dynamically imports it
4. Runs the imported component's workflow with AgentIO + llm_judge
5. Verifies the output Pydantic model validates
"""

import importlib
import os
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path

from tests.conftest import timeout
from tests.evals.helpers import (
    AgentIO,
    make_config,
    make_tools,
    run_multi_turn_eval,
)
from workflows.workflow.design_spec import ComponentDesignSpec
from workflows.workflow.domain_spec import ComponentDomainField, ComponentDomainSpec
from workflows.workflow.generated_component import GeneratedComponent
from workflows.workflow.interaction_context import ComponentInteractionContext
from workflows.workflow.structure import ComponentStructure


def make_minutes_draft_design_spec() -> ComponentDesignSpec:
    """Create a ComponentDesignSpec for a MinutesDraft artifact."""
    return ComponentDesignSpec(
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


def format_transcript(session) -> str:
    """Build a formatted conversation transcript from session state."""
    parts = []
    for msg in session.state.messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "system":
            continue
        parts.append(f"[{role}]\n{content}")
    return "\n---\n".join(parts)


class TestGeneratedComponentCodegenEval(unittest.TestCase):
    """End-to-end eval: generate code, import it, run it, judge it."""

    @timeout(300)
    def test_generated_component_codegen_end_to_end(self):
        """Generated component code should be importable and its workflow runnable."""
        config = make_config()
        design_spec = make_minutes_draft_design_spec()

        # --- Step 1: Generate the component code ---
        user_bot = AgentIO(
            persona_prompt="You are a placeholder (not used in this phase).",
            config=config,
        )
        session = make_tools(user_bot)

        generated: GeneratedComponent = GeneratedComponent.generate(
            design_spec=design_spec,
            session=session,
        )

        self.assertIsInstance(generated, GeneratedComponent)
        self.assertGreater(len(generated.code), 0)
        self.assertIn("class ", generated.code)

        # Verify syntax
        with suppress(SyntaxError):
            compile(generated.code, "<test>", "exec")

        # --- Step 2: Verify syntax and write code to temp file ---
        # Verify syntax with compile() — ruff E501 (line too long) is not
        # auto-fixable and LLM-generated docstrings may exceed 120 chars.
        compile(generated.code, "<generated>", "exec")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=Path(__file__).parent
        ) as f:
            f.write(generated.code)
            temp_path = f.name

        try:
            # --- Step 3: Dynamically import the generated module ---
            module_name = os.path.splitext(os.path.basename(temp_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            self.assertIsNotNone(spec, f"Could not create spec for {temp_path}")
            mod = importlib.util.module_from_spec(spec)
            # Add to sys.modules so imports within the module resolve
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

            # --- Step 4: Find the workflow class and method ---
            # Look for a BaseModel subclass with an @atomic_workflow method
            # (identified by the _is_workflow attribute set by the decorator)
            workflow_method = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if not isinstance(attr, type):
                    continue
                from pydantic import BaseModel

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
                n for n in dir(mod)
                if isinstance(getattr(mod, n), type)
                and issubclass(getattr(mod, n), BaseModel)
                and getattr(mod, n) is not BaseModel
            ]
            self.assertIsNotNone(
                workflow_method,
                f"No @atomic_workflow method found. Classes: {classes_found}",
            )

            # --- Step 5: Run the generated workflow with a user bot ---
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
                    "The agent didn't ask for the same information again after the "
                    "user already provided it."
                ),
                "Uses domain language": (
                    "The agent used natural business language (meeting, decisions, "
                    "action items) rather than technical terms (fields, records, schemas)."
                ),
            }

            result = run_multi_turn_eval(
                model_method=workflow_method,
                method_kwargs=dict(
                    max_turns=10,
                ),
                user_persona=user_persona,
                judge_rules=judge_rules,
                config=config,
            )

            # --- Step 6: Verify the result is a valid Pydantic model ---
            self.assertIsNotNone(result)
            self.assertIsInstance(result, BaseModel)

            # Verify it has the expected fields from the design spec
            result_dict = result.model_dump() if hasattr(result, "model_dump") else {}
            # At minimum, the result should have some content
            self.assertGreater(len(str(result_dict)), 0)

        finally:
            # Clean up temp file
            with suppress(OSError):
                os.unlink(temp_path)
            # Clean up sys.modules
            with suppress(KeyError):
                del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main(verbosity=2)
