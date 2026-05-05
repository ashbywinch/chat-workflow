"""
Configuration management for prompt-core.
Reads from config.json and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration manager for LLM settings."""

    _instance = None
    _config_data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from config.json and environment variables."""
        from .exceptions import ConfigFileError, ConfigurationError

        config_path = Path(__file__).parent.parent / "config.json"

        # Default configuration - no defaults for provider/model, they must be configured
        self._config_data = {
            "llm": {
                "temperature": 0.7,
                "max_retries": 3,
                "request_timeout_seconds": 30,
                "model_supports_tools": False,
            }
        }

        # config.json MUST exist
        if not config_path.exists():
            raise ConfigFileError(
                f"Configuration file not found: {config_path}\n"
                f"Create config.json from config.json.example: cp config.json.example config.json"
            )

        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                self._merge_config(self._config_data, file_config)
        except json.JSONDecodeError as e:
            raise ConfigFileError(f"Invalid JSON in config.json: {e}")
        except IOError as e:
            raise ConfigFileError(f"Could not read config.json: {e}")

        # Validate required fields
        if not self._config_data.get("llm", {}).get("provider"):
            raise ConfigurationError(
                "Missing required field in config.json: llm.provider"
            )

        if not self._config_data.get("llm", {}).get("model"):
            raise ConfigurationError("Missing required field in config.json: llm.model")

    def _merge_config(self, base: Dict[str, Any], overlay: Dict[str, Any]):
        """Recursively merge overlay configuration into base."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    @property
    def provider(self) -> str:
        """Get the configured LLM provider."""
        return self._config_data["llm"]["provider"]

    @property
    def model(self) -> str:
        """Get the configured LLM model."""
        return self._config_data["llm"]["model"]

    @property
    def temperature(self) -> float:
        """Get the configured temperature."""
        return self._config_data["llm"]["temperature"]

    @property
    def max_retries(self) -> int:
        """Get the configured max retries."""
        return self._config_data["llm"]["max_retries"]

    @property
    def request_timeout_seconds(self) -> float:
        """Get the configured timeout for each LLM request."""
        return self._config_data["llm"]["request_timeout_seconds"]

    @property
    def model_supports_tools(self) -> bool:
        """Whether the configured model supports tool/function calling."""
        return self._config_data["llm"].get("model_supports_tools", False)

    @property
    def debug(self) -> bool:
        """Whether to enable debug tracing of LLM interactions."""
        return os.environ.get("PROMPT_CORE_DEBUG", "").lower() in ("1", "true", "yes")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key."""
        keys = key.split(".")
        value = self._config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self._config_data, indent=2)


# Global configuration instance
config = Config()
