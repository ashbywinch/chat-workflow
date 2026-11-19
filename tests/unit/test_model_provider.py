"""Provider resolution: one place maps a config's provider to a model."""

import unittest
from unittest.mock import patch

from chat_workflow.exceptions import APIKeyError, ProviderNotSupportedError
from chat_workflow.model_provider import build_model, list_available_providers


class TestBuildModel(unittest.TestCase):
    def test_an_unknown_provider_is_refused_by_name(self):
        with self.assertRaises(ProviderNotSupportedError) as ctx:
            build_model(provider="nonsense", model="m")

        self.assertIn("nonsense", str(ctx.exception))

    def test_google_and_anthropic_name_the_extra_to_install(self):
        for provider in ("google", "anthropic"):
            with self.subTest(provider=provider), self.assertRaises(ProviderNotSupportedError) as ctx:
                build_model(provider=provider, model="m")

            self.assertIn(f"pydantic-ai-slim[{provider}]", str(ctx.exception))

    def test_a_missing_key_is_a_configuration_error(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(APIKeyError) as ctx:
            build_model(provider="openai", model="m")

        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_the_api_key_env_override_is_read(self):
        with patch.dict("os.environ", {"MY_KEY": "secret"}, clear=True):
            model = build_model(provider="openai", model="m", api_key_env="MY_KEY")

        self.assertEqual(model.model_name, "m")

    def test_the_gateway_base_url_is_used(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            model = build_model(
                provider="openai", model="route/model", api_base="https://gateway.example/compat"
            )

        assert model.provider is not None
        self.assertIn("gateway.example", model.provider.base_url)
        self.assertEqual(model.model_name, "route/model")

    def test_a_prefix_naming_the_provider_is_not_part_of_the_model_id(self):
        """config.json spells a route ``openai/dynamic/fallback2``; the wire wants the id.

        A gateway reads the leading ``openai/`` as "forward to OpenAI with the
        caller's key", which loses the gateway's own auth and returns OpenAI's
        401. ``model_name`` is the string the SDK sends as ``model``.
        """
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            model = build_model(provider="openai", model="openai/dynamic/fallback2")

        self.assertEqual(model.model_name, "dynamic/fallback2")

    def test_another_vendors_prefix_stays_in_the_model_id(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}, clear=True):
            model = build_model(provider="openrouter", model="openai/gpt-4o")

        self.assertEqual(model.model_name, "openai/gpt-4o")

    def test_openrouter_uses_its_published_host(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}, clear=True):
            model = build_model(provider="openrouter", model="vendor/model")

        assert model.provider is not None
        self.assertIn("openrouter.ai", model.provider.base_url)

    def test_azure_reads_its_endpoint(self):
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com",
                "OPENAI_API_VERSION": "2024-10-21",
            },
            clear=True,
        ):
            model = build_model(provider="azure", model="deployment")

        self.assertEqual(getattr(model.provider, "azure_endpoint", None), "https://res.openai.azure.com")


class TestListProviders(unittest.TestCase):
    def test_all_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(any(list_available_providers().values()))

    def test_one_set(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            providers = list_available_providers()

        self.assertTrue(providers["openai"])
        self.assertFalse(providers["google"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
