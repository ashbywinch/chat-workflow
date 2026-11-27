"""Tests for the prompt builder: parameters, task, checklist and rules."""

import unittest
from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow.annotations import Validation
from chat_workflow.decorators import _build_system_prompt
from chat_workflow.mixins import LLMValidated
from chat_workflow.prompt_builder import build_params_section


class InnerShape(BaseModel):
    """A nested input shape."""

    name: str = Field(..., description="The name of the thing", min_length=1)


def example_workflow(
    inner: Annotated[InnerShape, "The inner shape"],
    max_turns: Annotated[int, "Maximum conversation turns"] = 3,
):
    """Do the example task."""


class TestBuildParamsSection(unittest.TestCase):
    def test_includes_pydantic_model_shape(self):
        section = build_params_section(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
        )

        self.assertIn("The name of the thing", section)
        self.assertIn("minLength", section)
        self.assertIn("Shape", section)

    def test_skips_shape_already_in_output_schema(self):
        section = build_params_section(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            skip_shapes=frozenset({"InnerShape"}),
        )

        self.assertNotIn("  Shape:", section)
        self.assertIn("The inner shape", section)


class TestBuildSystemPrompt(unittest.TestCase):
    def test_includes_procedure_checklist_and_object_rules(self):
        prompt = _build_system_prompt(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            return_type=InnerShape,
            conversation_rules={"No repetition": "Do not repeat."},
            chat=True,
        )

        self.assertIn("Do the example task.", prompt)
        self.assertIn("## Conversation checklist", prompt)
        self.assertIn("No repetition", prompt)
        self.assertNotIn("## Task", prompt)

    def test_session_instructions_come_first(self):
        prompt = _build_system_prompt(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            return_type=InnerShape,
            conversation_rules={"No repetition": "Do not repeat."},
            chat=True,
            instructions="Hold one conversation about the user's work.",
        )

        self.assertTrue(
            prompt.startswith("Hold one conversation about the user's work.")
        )
        self.assertLess(
            prompt.index("Hold one conversation"),
            prompt.index("Do the example task."),
        )

    def test_chat_off_omits_conversation_checklist(self):
        prompt = _build_system_prompt(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            return_type=InnerShape,
            conversation_rules={"No repetition": "Do not repeat."},
            chat=False,
        )

        self.assertIn("Do the example task.", prompt)
        self.assertNotIn("## Conversation checklist", prompt)
        self.assertNotIn("## Task", prompt)


class TestNoProtocolText(unittest.TestCase):
    """The schema owns the protocol; an authored prompt never restates it."""

    def test_prompt_does_not_state_the_response_protocol(self):
        prompt = _build_system_prompt(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            return_type=InnerShape,
            conversation_rules={"No repetition": "Do not repeat."},
            chat=True,
        )

        lowered = prompt.lower()
        for phrase in (
            "return json",
            "put the user-facing text in `message`",
            "never reply with plain text",
            "set `intent`",
        ):
            self.assertNotIn(phrase, lowered)

class TestValidationSchema(unittest.TestCase):
    def test_validation_rule_reaches_json_schema(self):
        class Validated(LLMValidated):
            x: Annotated[str, Validation("must be nonempty")] = Field(
                ..., description="x"
            )

        description = Validated.model_json_schema()["properties"]["x"]["description"]
        self.assertIn("must be nonempty", description)


if __name__ == "__main__":
    unittest.main(verbosity=2)
