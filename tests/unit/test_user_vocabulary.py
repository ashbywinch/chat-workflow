"""User-facing labels keep the result schema out of the assistant's message."""

import unittest
from typing import Annotated

from pydantic import BaseModel

from chat_workflow.annotations import UserLabel, user_labels
from chat_workflow.atomic_workflow_step import AtomicWorkflowStep
from chat_workflow.decorators import _build_user_vocabulary_section
from chat_workflow.models import AgentResponse
from tests.conftest import make_atomic_workflow_step_config


class LabelledResult(BaseModel):
    success_criteria: Annotated[
        str,
        UserLabel("how you know it is good", "success criteria", "criteria"),
    ]


def _step() -> AtomicWorkflowStep:
    config = make_atomic_workflow_step_config(
        response_model_override=AgentResponse[LabelledResult],
        system_prompt="Test prompt",
    )
    return AtomicWorkflowStep(config=config)


def _continue(message: str) -> AgentResponse[LabelledResult]:
    return AgentResponse[LabelledResult](intent="continue", message=message)


class TestUserLabelExtraction(unittest.TestCase):
    def test_label_sits_on_the_field(self):
        labels = user_labels(LabelledResult)

        self.assertEqual(labels["success_criteria"].phrase, "how you know it is good")
        self.assertIn("criteria", labels["success_criteria"].aliases)


class TestVocabularyPrompt(unittest.TestCase):
    def test_section_maps_internal_label_to_user_phrase(self):
        section = _build_user_vocabulary_section(LabelledResult)

        self.assertIn("success_criteria", section)
        self.assertIn("how you know it is good", section)
        self.assertIn("user used them first", section)


if __name__ == "__main__":
    unittest.main(verbosity=2)
