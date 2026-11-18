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


class LabelledInput(BaseModel):
    orchestrating_component: Annotated[
        str,
        UserLabel("who keeps things moving", "orchestrating component", "orchestrating"),
    ]
    phases: Annotated[
        list[str],
        UserLabel("the main stages", "phases", guard=False),
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
        self.assertIn("use the phrase below", section)

    def test_input_labels_get_the_same_phrases(self):
        """Input objects reach the model as shapes, so their labels must too."""
        section = _build_user_vocabulary_section(LabelledResult, (LabelledInput,))

        self.assertIn("orchestrating_component", section)
        self.assertIn("who keeps things moving", section)
        self.assertIn("the main stages", section)
        self.assertIn("internal engineering text", section)

    def test_parameter_and_type_names_get_the_same_treatment(self):
        """A parameter name and a model name are internal too, not user words."""
        section = _build_user_vocabulary_section(
            LabelledResult,
            (LabelledInput,),
            {"analysis": "What the person already told you"},
        )

        self.assertIn("`analysis` -> \"What the person already told you\"", section)
        self.assertIn("`LabelledInput`", section)
        self.assertIn("`LabelledResult`", section)
        self.assertIn("are internal names for it", section)

    def test_input_models_are_read_from_the_signature(self):
        from chat_workflow.prompt_builder import input_models, user_parameter_labels

        def step(
            analysis: Annotated[LabelledInput, "What the person told you"],
            max_turns: int = 10,
        ) -> LabelledResult:
            ...

        self.assertEqual(input_models(step), (LabelledInput,))
        self.assertEqual(
            user_parameter_labels(step),
            {"analysis": "What the person told you"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
