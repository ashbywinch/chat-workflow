"""Configuration management for chat-workflow. Reads from config.json."""

import json
import os
from pathlib import Path
from typing import Any

from .llm_interaction import ProviderType


class Config:
    """Configuration manager for LLM settings.

    The caller provides the path to config.json. No auto-discovery.
    """

    _VALID_PROVIDERS = frozenset({"openai", "google", "anthropic", "groq", "together", "azure", "openrouter"})

    def __init__(self, config_path: Path):
        from .exceptions import ConfigFileError, ConfigurationError

        self._config_data: dict[str, Any] = {
            "llm": {
                "temperature": 0.7,
                "max_retries": 3,
                "request_timeout_seconds": 30,
                "model_supports_tools": False,
            }
        }

        if not config_path.exists():
            raise ConfigFileError(
                f"Configuration file not found: {config_path}\nRefer to config.json.example for the required format."
            )

        try:
            with open(config_path) as f:
                file_config = json.load(f)
                self._merge_config(self._config_data, file_config)
        except json.JSONDecodeError as e:
            raise ConfigFileError(f"Invalid JSON in {config_path}: {e}") from e
        except OSError as e:
            raise ConfigFileError(f"Could not read {config_path}: {e}") from e

        if not self._config_data.get("llm", {}).get("provider"):
            raise ConfigurationError("Missing required field in config.json: llm.provider")

        provider = self._config_data["llm"]["provider"]
        if provider not in self._VALID_PROVIDERS:
            raise ConfigurationError(
                f"Invalid provider in config.json: {provider}. "
                f"Valid providers: {', '.join(sorted(self._VALID_PROVIDERS))}"
            )

        if not self._config_data.get("llm", {}).get("model"):
            raise ConfigurationError("Missing required field in config.json: llm.model")

    def _merge_config(self, base: dict[str, Any], overlay: dict[str, Any]):
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    @property
    def provider(self) -> "ProviderType":
        return self._config_data["llm"]["provider"]

    @property
    def model(self) -> str:
        return self._config_data["llm"]["model"]

    @property
    def temperature(self) -> float:
        return self._config_data["llm"]["temperature"]

    @property
    def max_retries(self) -> int:
        return self._config_data["llm"]["max_retries"]

    @property
    def request_timeout_seconds(self) -> float:
        return self._config_data["llm"]["request_timeout_seconds"]

    @property
    def model_supports_tools(self) -> bool:
        return self._config_data["llm"].get("model_supports_tools", False)

    @property
    def api_base(self) -> str | None:
        """Custom API base URL (e.g. https://opencode.ai/zen/go/v1). None = provider default."""
        return self._config_data["llm"].get("api_base")

    @property
    def api_key_env(self) -> str | None:
        """Environment variable holding the API key. None = provider default (e.g. OPENAI_API_KEY)."""
        return self._config_data["llm"].get("api_key_env")

    @property
    def debug(self) -> bool:
        return os.environ.get("CHAT_WORKFLOW_DEBUG", "").lower() in ("1", "true", "yes")

    def __str__(self) -> str:
        return json.dumps(self._config_data, indent=2)
