"""Configuration management for chat-workflow. Reads from config.json."""

import json
import os
from pathlib import Path
from typing import Any


class Config:
    """Configuration manager for LLM settings.

    The caller provides the path to config.json. No auto-discovery.
    """

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
                f"Configuration file not found: {config_path}\n"
                f"Refer to config.json.example for the required format."
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
            raise ConfigurationError(
                "Missing required field in config.json: llm.provider"
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
    def provider(self) -> str:
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
    def debug(self) -> bool:
        return os.environ.get("CHAT_WORKFLOW_DEBUG", "").lower() in ("1", "true", "yes")

    def __str__(self) -> str:
        return json.dumps(self._config_data, indent=2)
