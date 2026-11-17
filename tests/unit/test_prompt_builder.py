"""Tests for the prompt builder: parameters, task, checklist and rules."""

import unittest
from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow.annotations import Validation
from chat_workflow.decorators import _build_system_prompt
from chat_workflow.mixins import LLMValidated
from chat_workflow.prompt_builder import _build_params_section


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
        section = _build_params_section(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
        )

        self.assertIn("The name of the thing", section)
        self.assertIn("minLength", section)
        self.assertIn("Shape", section)

    def test_skips_shape_already_in_output_schema(self):
        section = _build_params_section(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            skip_shapes=frozenset({"InnerShape"}),
        )

        self.assertNotIn("  Shape:", section)
        self.assertIn("The inner shape", section)


class TestBuildSystemPrompt(unittest.TestCase):
    def test_includes_task_checklist_and_object_rules(self):
        prompt = _build_system_prompt(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            return_type=InnerShape,
            conversation_rules={"No repetition": "Do not repeat."},
            chat=True,
        )

        self.assertIn("## Task", prompt)
        self.assertIn("InnerShape", prompt)
        self.assertIn("## Conversation checklist", prompt)
        self.assertIn("No repetition", prompt)

    def test_chat_off_omits_conversation_checklist(self):
        prompt = _build_system_prompt(
            example_workflow,
            {"inner": InnerShape(name="x"), "max_turns": 3},
            return_type=InnerShape,
            conversation_rules={"No repetition": "Do not repeat."},
            chat=False,
        )

        self.assertIn("## Task", prompt)
        self.assertNotIn("## Conversation checklist", prompt)


class TestChatTaskSection(unittest.TestCase):
    """The Task section states the protocol; the checklist states the rules."""

    def task(self) -> str:
        from chat_workflow.decorators import _build_task_section

        return _build_task_section(InnerShape, chat=True)

    def test_answering_comes_before_success(self):
        """A success turn carries no message, so the answer is its own turn."""
        task = self.task()

        self.assertIn("answer it in a continue turn", task)
        self.assertIn("after their next turn", task)
        self.assertNotIn("then return the result in the same response", task)

    def test_rules_are_not_restated(self):
        """Each conversation rule has one statement: the checklist."""
        task = self.task()

        self.assertNotIn("Ask each question once", task)
        self.assertNotIn("thinking partner", task.lower())
        self.assertNotIn("Your input parameters", task)


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
