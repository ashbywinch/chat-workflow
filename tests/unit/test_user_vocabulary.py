"""User-facing labels keep the result schema out of the assistant's message."""

import unittest
from typing import Annotated

from chat_workflow.annotations import UserLabel, user_labels
from chat_workflow.decorators import _build_user_vocabulary_section
from chat_workflow.mixins import LLMValidated


class LabelledResult(LLMValidated):
    success_criteria: Annotated[
        str,
        UserLabel(phrase="how you know it is good", aliases=("success criteria", "criteria")),
    ]


class LabelledInput(LLMValidated):
    orchestrating_component: Annotated[
        str,
        UserLabel(phrase="who keeps things moving", aliases=("orchestrating component", "orchestrating")),
    ]
    phases: Annotated[
        list[str],
        UserLabel(phrase="the main stages", aliases="phases", guard=False),
    ]


class TestUserLabelExtraction(unittest.TestCase):
    def test_label_sits_on_the_field(self):
        labels = user_labels(LabelledResult)

        self.assertEqual(labels["success_criteria"].phrase, "how you know it is good")
        self.assertIn("criteria", labels["success_criteria"].aliases)


class TestVocabularyPrompt(unittest.TestCase):
    def test_the_phrase_is_the_schema_title_of_the_result_field(self):
        """The label lives on the class and is rendered where the model writes."""
        title = LabelledResult.model_json_schema()["properties"]["success_criteria"]["title"]

        self.assertEqual(title, "how you know it is good")

    def test_the_phrase_is_the_schema_title_of_an_input_field(self):
        titles = LabelledInput.model_json_schema()["properties"]

        self.assertEqual(titles["orchestrating_component"]["title"], "who keeps things moving")
        self.assertEqual(titles["phases"]["title"], "the main stages")

    def test_the_section_points_at_the_field_title(self):
        section = _build_user_vocabulary_section(LabelledResult)

        self.assertIn("Each field's title is the phrase", section)
        self.assertIn("internal engineering text", section)
        self.assertNotIn("success_criteria", section)
        self.assertNotIn("`message`", section)

    def test_parameter_and_type_names_get_the_same_treatment(self):
        """A parameter name and a model name are internal too, not user words."""
        section = _build_user_vocabulary_section(
            LabelledResult,
            (LabelledInput,),
            {"analysis": "What the person already told you"},
        )

        self.assertIn("`analysis` -> \"What the person already told you\"", section)
        self.assertIn("type names above are internal", section)

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
