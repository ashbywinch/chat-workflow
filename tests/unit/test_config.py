"""Tests for chat_workflow.config: Config reads from a caller-provided path."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from chat_workflow import Config


class TestConfig(unittest.TestCase):
    """Config requires an explicit path to config.json — no auto-discovery."""

    def _write_temp_config(self, content: object) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            return f.name

    def test_missing_config_file_raises_error(self):
        from chat_workflow import ConfigFileError

        with self.assertRaises(ConfigFileError):
            Config(Path("/nonexistent/config.json"))

    def test_invalid_json_raises_error(self):
        from chat_workflow import ConfigFileError

        temp_path = tempfile.mktemp(suffix=".json")
        try:
            with open(temp_path, "w") as f:
                f.write("not valid json {{{")
            with self.assertRaises(ConfigFileError):
                Config(Path(temp_path))
        finally:
            os.unlink(temp_path)

    def test_io_error_raises_config_file_error(self):
        from chat_workflow import ConfigFileError

        temp_path = tempfile.mktemp(suffix=".json")
        try:
            with open(temp_path, "w") as f:
                json.dump({"llm": {"provider": "x", "model": "y"}}, f)
            os.chmod(temp_path, 0o000)
            with self.assertRaises(ConfigFileError):
                Config(Path(temp_path))
        finally:
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)

    def test_missing_provider_raises_error(self):
        from chat_workflow import ConfigurationError

        temp_path = self._write_temp_config({"llm": {"model": "gpt-4"}})
        try:
            with self.assertRaises(ConfigurationError):
                Config(Path(temp_path))
        finally:
            os.unlink(temp_path)

    def test_missing_model_raises_error(self):
        from chat_workflow import ConfigurationError

        temp_path = self._write_temp_config({"llm": {"provider": "openai"}})
        try:
            with self.assertRaises(ConfigurationError):
                Config(Path(temp_path))
        finally:
            os.unlink(temp_path)

    def test_empty_provider_raises_error(self):
        from chat_workflow import ConfigurationError

        temp_path = self._write_temp_config({"llm": {"provider": "", "model": "gpt-4"}})
        try:
            with self.assertRaises(ConfigurationError):
                Config(Path(temp_path))
        finally:
            os.unlink(temp_path)

    def test_empty_model_raises_error(self):
        from chat_workflow import ConfigurationError

        temp_path = self._write_temp_config({"llm": {"provider": "openai", "model": ""}})
        try:
            with self.assertRaises(ConfigurationError):
                Config(Path(temp_path))
        finally:
            os.unlink(temp_path)

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
                "model_supports_tools": True,
            }
        }
        temp_path = self._write_temp_config(data)
        try:
            cfg = Config(Path(temp_path))
            self.assertEqual(cfg.provider, "openai")
            self.assertEqual(cfg.model, "openai/deepseek-v4-flash")
            self.assertEqual(cfg.api_base, "https://opencode.ai/zen/go/v1")
            self.assertEqual(cfg.api_key_env, "OPENCODE_GO_EVALS_API_KEY")
            self.assertEqual(cfg.temperature, 0.3)
            self.assertEqual(cfg.max_retries, 5)
            self.assertEqual(cfg.request_timeout_seconds, 60)
            self.assertTrue(cfg.model_supports_tools)
        finally:
            os.unlink(temp_path)

    def test_api_base_and_key_env_default_to_none(self):
        temp_path = self._write_temp_config({"llm": {"provider": "openai", "model": "gpt-4"}})
        try:
            cfg = Config(Path(temp_path))
            self.assertIsNone(cfg.api_base)
            self.assertIsNone(cfg.api_key_env)
        finally:
            os.unlink(temp_path)

    def test_default_values_applied(self):
        temp_path = self._write_temp_config({"llm": {"provider": "anthropic", "model": "claude-3-opus"}})
        try:
            cfg = Config(Path(temp_path))
            self.assertEqual(cfg.temperature, 0.7)
            self.assertEqual(cfg.max_retries, 3)
            self.assertEqual(cfg.request_timeout_seconds, 30)
            self.assertFalse(cfg.model_supports_tools)
        finally:
            os.unlink(temp_path)

    def test_partial_overrides(self):
        temp_path = self._write_temp_config(
            {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "request_timeout_seconds": 120,
                }
            }
        )
        try:
            cfg = Config(Path(temp_path))
            self.assertEqual(cfg.temperature, 0.5)
            self.assertEqual(cfg.max_retries, 3)
            self.assertEqual(cfg.request_timeout_seconds, 120)
            self.assertFalse(cfg.model_supports_tools)
        finally:
            os.unlink(temp_path)

    def test_str_representation(self):
        temp_path = self._write_temp_config({"llm": {"provider": "anthropic", "model": "test-m"}})
        try:
            cfg = Config(Path(temp_path))
            s = str(cfg)
            self.assertIn("anthropic", s)
            self.assertIn("test-m", s)
        finally:
            os.unlink(temp_path)

    def test_debug_from_env(self):
        temp_path = self._write_temp_config({"llm": {"provider": "anthropic", "model": "test-m"}})
        try:
            cfg = Config(Path(temp_path))
            self.assertFalse(cfg.debug)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
