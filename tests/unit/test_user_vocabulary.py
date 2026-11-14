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


class TestVocabularyGuard(unittest.TestCase):
    def test_internal_label_is_rejected_before_the_user_uses_it(self):
        step = _step()

        failure = step._vocabulary_failure(_continue("Tell me the criteria."))

        assert failure is not None
        self.assertIn("internal label 'criteria'", failure.reason)
        self.assertIn("how you know it is good", failure.message())

    def test_the_user_word_is_allowed(self):
        step = _step()
        step.messages.append({"role": "user", "content": "I care about criteria."})

        self.assertIsNone(step._vocabulary_failure(_continue("Tell me the criteria.")))

    def test_the_phrase_is_not_an_internal_label(self):
        step = _step()

        self.assertIsNone(
            step._vocabulary_failure(_continue("Tell me how you know it is good."))
        )

    def test_framework_word_is_rejected_before_the_user_uses_it(self):
        step = _step()

        failure = step._vocabulary_failure(_continue("Let's map your workflow."))

        assert failure is not None
        self.assertIn("internal label 'workflow'", failure.reason)

    def test_framework_word_is_allowed_when_the_user_uses_it(self):
        step = _step()
        step.messages.append({"role": "user", "content": "I run a workflow."})

        self.assertIsNone(step._vocabulary_failure(_continue("Let's map your workflow.")))


class TestRepetitionGuard(unittest.TestCase):
    def test_confirmation_question_after_yes_is_rejected(self):
        step = _step()
        step.messages.append({"role": "user", "content": "Yes, exactly."})

        failure = step._repetition_failure(
            _continue("Does that match your picture?")
        )

        assert failure is not None
        self.assertIn("already confirmed", failure.reason)

    def test_a_new_question_after_yes_is_allowed(self):
        step = _step()
        step.messages.append({"role": "user", "content": "Yes, exactly."})

        self.assertIsNone(
            step._repetition_failure(_continue("What should I do next?"))
        )

    def test_yes_with_a_caveat_is_not_a_confirmation(self):
        step = _step()
        step.messages.append({"role": "user", "content": "Yes, but I have one change."})

        self.assertIsNone(
            step._repetition_failure(_continue("Does that match your picture?"))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
