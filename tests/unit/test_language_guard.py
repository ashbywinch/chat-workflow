"""Unit tests for the deterministic language-drift guard.

The eval suite caught a Chinese reply in an English conversation. The model
cannot be relied on to obey the prompt, so the framework rejects a reply in
the wrong script and feeds the correction back like any other validation
failure.
"""

import unittest

from chat_workflow.atomic_workflow_step import (
    _is_wrong_language,
    _language_reference,
)


class TestLanguageGuard(unittest.TestCase):
    def test_chinese_reply_to_english_is_wrong(self):
        self.assertTrue(
            _is_wrong_language(
                "Help me plan my meals.",
                "我来帮你整理一下产物、输入和所有者。",
            )
        )

    def test_english_reply_is_fine(self):
        self.assertFalse(
            _is_wrong_language(
                "Help me plan my meals.",
                "Let's describe the meal plan you make.",
            )
        )

    def test_chinese_conversation_is_left_alone(self):
        self.assertFalse(
            _is_wrong_language(
                "帮我整理一下我的会议记录流程。",
                "好的，我先看看你说的主要内容。",
            )
        )

    def test_a_quoted_name_is_not_drift(self):
        self.assertFalse(
            _is_wrong_language(
                "Help me plan my meals.",
                "You called it 会议记录; let's describe that.",
            )
        )

    def test_last_real_user_turn_sets_the_language(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Help me plan my meals."},
            {"role": "assistant", "content": "Sure."},
            {"role": "user", "content": "[framework] Fix this violation."},
        ]

        self.assertEqual(_language_reference(messages), "Help me plan my meals.")

    def test_system_prompt_is_the_reference_before_the_first_turn(self):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        self.assertEqual(_language_reference(messages), "You are a helpful assistant.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
