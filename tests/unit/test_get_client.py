"""Tests for chat_workflow.llm_interaction.get_client — provider resolution."""

import unittest
from unittest.mock import patch


class TestGetClient(unittest.TestCase):
    """get_client validates provider and reads API key from env."""

    def test_unsupported_provider_raises_error(self):
        from chat_workflow.exceptions import ProviderNotSupportedError
        from chat_workflow.llm_interaction import get_client

        with self.assertRaises(ProviderNotSupportedError):
            get_client("nonexistent")

    def test_missing_api_key_raises_error(self):
        from chat_workflow.exceptions import APIKeyError
        from chat_workflow.llm_interaction import get_client

        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaises(APIKeyError)
        ):
            get_client("openai")

    def test_provider_case_insensitive(self):
        from chat_workflow.llm_interaction import get_client

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("chat_workflow.llm_interaction.instructor") as mock_instructor
        ):
            mock_instructor.from_litellm.return_value = "client"
            result = get_client("OpenAI")
            self.assertEqual(result, "client")

    def test_multiple_providers(self):
        from chat_workflow.llm_interaction import get_client

        providers = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        for provider_name, env_var in providers.items():
            with (
                self.subTest(provider=provider_name),
                patch.dict("os.environ", {env_var: "test-key"}, clear=True),
                patch(
                    "chat_workflow.llm_interaction.instructor"
                ) as mock_instructor
            ):
                mock_instructor.from_litellm.return_value = "client"
                result = get_client(provider_name)
                self.assertEqual(result, "client")


class TestListAvailableProviders(unittest.TestCase):

    def test_all_unset(self):
        from chat_workflow.llm_interaction import list_available_providers

        with patch.dict("os.environ", {}, clear=True):
            result = list_available_providers()
            self.assertFalse(any(result.values()))

    def test_one_set(self):
        from chat_workflow.llm_interaction import list_available_providers

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = list_available_providers()
            self.assertTrue(result["openai"])
            self.assertFalse(result["google"])
