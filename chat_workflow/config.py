"""Configuration management for chat-workflow. Reads from config.json."""

import json
import os
from pathlib import Path
from typing import Any

from .model_provider import PROVIDERS, ProviderType


class Config:
    """Configuration manager for LLM settings.

    The caller provides the path to config.json. No auto-discovery.
    ``overrides`` is merged over the file, for an application that varies a
    setting (an eval that must bypass a gateway cache) without a second file.
    """

    _VALID_PROVIDERS = PROVIDERS

    def __init__(self, config_path: Path, overrides: dict[str, Any] | None = None):
        from .exceptions import ConfigFileError, ConfigurationError

        self._config_data: dict[str, Any] = {
            "llm": {
                "temperature": 0.7,
                "max_retries": 3,
                "request_timeout_seconds": 30,
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

        if overrides:
            self._merge_config(self._config_data, overrides)

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
    def extra_headers(self) -> dict[str, str]:
        """Headers sent with every model request.

        A gateway may cache responses by request body, which makes an eval
        replay an old answer instead of testing the current prompt. Set
        ``"extra_headers": {"cf-aig-skip-cache": "true"}`` to bypass it.
        """
        return dict(self._config_data["llm"].get("extra_headers") or {})

    @property
    def api_base(self) -> str | None:
        """The endpoint the configured provider talks to.

        A stated base wins. The ``openai`` provider without one is pinned to
        ``OPENAI_BASE_URL`` — the house gateway — so an ambient base cannot
        route the call to another host. An OpenAI-compatible provider with no
        stated base uses its published host (see ``model_provider``).
        """
        stated = self._config_data["llm"].get("api_base")
        if stated:
            return str(stated)
        if self.provider == "openai":
            return os.environ.get("OPENAI_BASE_URL") or None
        return None

    @property
    def api_key_env(self) -> str | None:
        """Environment variable holding the API key. None = provider default (e.g. OPENAI_API_KEY)."""
        return self._config_data["llm"].get("api_key_env")

    @property
    def debug(self) -> bool:
        return os.environ.get("CHAT_WORKFLOW_DEBUG", "").lower() in ("1", "true", "yes")

    def __str__(self) -> str:
        return json.dumps(self._config_data, indent=2)
