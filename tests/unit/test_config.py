"""Tests for chat_workflow.config: Config reads from a caller-provided path."""

import json
import os
import unittest
from pathlib import Path
from typing import override

from pyfakefs.fake_filesystem_unittest import TestCase

from chat_workflow import Config

CONFIG_PATH = Path("/config/config.json")


class TestConfig(TestCase):
    """Config requires an explicit path to config.json — no auto-discovery."""

    @override
    def setUp(self):
        self.setUpPyfakefs()

    def _write_config(self, content: object, path: Path = CONFIG_PATH) -> Path:
        """Put *content* in the fake filesystem and return the path to it."""
        self.fs.create_file(path, contents=json.dumps(content))
        return path

    def test_missing_config_file_raises_error(self):
        from chat_workflow import ConfigFileError

        with self.assertRaises(ConfigFileError):
            Config(Path("/nonexistent/config.json"))

    def test_invalid_json_raises_error(self):
        from chat_workflow import ConfigFileError

        self.fs.create_file(CONFIG_PATH, contents="not valid json {{{")

        with self.assertRaises(ConfigFileError):
            Config(CONFIG_PATH)

    def test_io_error_raises_config_file_error(self):
        from chat_workflow import ConfigFileError

        path = self._write_config({"llm": {"provider": "x", "model": "y"}})
        os.chmod(path, 0o000)

        with self.assertRaises(ConfigFileError):
            Config(path)

    def test_missing_provider_raises_error(self):
        from chat_workflow import ConfigurationError

        with self.assertRaises(ConfigurationError):
            Config(self._write_config({"llm": {"model": "gpt-4"}}))

    def test_missing_model_raises_error(self):
        from chat_workflow import ConfigurationError

        with self.assertRaises(ConfigurationError):
            Config(self._write_config({"llm": {"provider": "openai"}}))

    def test_empty_provider_raises_error(self):
        from chat_workflow import ConfigurationError

        with self.assertRaises(ConfigurationError):
            Config(self._write_config({"llm": {"provider": "", "model": "gpt-4"}}))

    def test_empty_model_raises_error(self):
        from chat_workflow import ConfigurationError

        with self.assertRaises(ConfigurationError):
            Config(self._write_config({"llm": {"provider": "openai", "model": ""}}))

    def test_all_properties_from_config(self):
        data = {
            "llm": {
                "provider": "openai",
                "model": "openai/deepseek-v4-flash",
                "api_base": "https://opencode.ai/zen/go/v1",
                "api_key_env": "OPENCODE_GO_EVALS_API_KEY",
                "temperature": 0.3,
                "max_retries": 5,
                "request_timeout_seconds": 60,
                "extra_headers": {"cf-aig-skip-cache": "true"},
            }
        }
        cfg = Config(self._write_config(data))

        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.model, "openai/deepseek-v4-flash")
        self.assertEqual(cfg.api_base, "https://opencode.ai/zen/go/v1")
        self.assertEqual(cfg.api_key_env, "OPENCODE_GO_EVALS_API_KEY")
        self.assertEqual(cfg.temperature, 0.3)
        self.assertEqual(cfg.max_retries, 5)
        self.assertEqual(cfg.request_timeout_seconds, 60)
        self.assertEqual(cfg.extra_headers, {"cf-aig-skip-cache": "true"})

    def test_api_base_and_key_env_default_to_none(self):
        cfg = Config(self._write_config({"llm": {"provider": "openai", "model": "gpt-4"}}))

        # The openai provider is pinned to the house gateway when no base
        # is stated; the key env is the provider default.
        self.assertEqual(cfg.api_base, os.environ.get("OPENAI_BASE_URL") or None)
        self.assertIsNone(cfg.api_key_env)

    def test_default_values_applied(self):
        cfg = Config(self._write_config({"llm": {"provider": "anthropic", "model": "claude-3-opus"}}))

        self.assertEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.max_retries, 3)
        self.assertEqual(cfg.request_timeout_seconds, 30)
        self.assertEqual(cfg.extra_headers, {})

    def test_partial_overrides(self):
        cfg = Config(
            self._write_config(
                {
                    "llm": {
                        "provider": "openai",
                        "model": "gpt-4",
                        "temperature": 0.5,
                        "request_timeout_seconds": 120,
                    }
                }
            )
        )

        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.max_retries, 3)
        self.assertEqual(cfg.request_timeout_seconds, 120)
        self.assertEqual(cfg.extra_headers, {})

    def test_str_representation(self):
        cfg = Config(self._write_config({"llm": {"provider": "anthropic", "model": "test-m"}}))

        s = str(cfg)
        self.assertIn("anthropic", s)
        self.assertIn("test-m", s)

    def test_debug_from_env(self):
        cfg = Config(self._write_config({"llm": {"provider": "anthropic", "model": "test-m"}}))

        self.assertFalse(cfg.debug)

    def test_overrides_merge_over_the_file(self):
        """An eval states 'do not serve me the cache' without a second config file."""
        cfg = Config(
            self._write_config({"llm": {"provider": "openai", "model": "m", "temperature": 0.2}}),
            overrides={"llm": {"extra_headers": {"cf-aig-skip-cache": "true"}}},
        )

        self.assertEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.extra_headers, {"cf-aig-skip-cache": "true"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
