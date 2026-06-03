"""Multi-provider LLM client via instructor and litellm."""

import os
from typing import Literal

import instructor
import litellm
from dotenv import load_dotenv
from litellm import completion

from .exceptions import (
    APIKeyError,
    ProviderNotSupportedError,
)

load_dotenv()
litellm.suppress_debug_info = True

ProviderType = Literal["openai", "google", "anthropic", "groq", "together", "azure", "openrouter"]

_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_client(
    provider: str,
    *,
    api_key_env: str | None = None,
    api_base: str | None = None,
    model_supports_tools: bool = False,
):
    """Get an instructor-patched LLM client for *provider*.

    The API key is read from the corresponding environment variable
    (e.g. ``OPENAI_API_KEY`` for ``provider="openai"``), unless
    *api_key_env* is given, in which case that env var is used instead.

    When *api_base* is given it is set as the provider-specific
    ``OPENAI_API_BASE`` (or ``<PROVIDER>_API_BASE``) environment
    variable so litellm routes requests to the custom endpoint.

    When *model_supports_tools* is True the client uses Instructor's
    native tool-calling mode (``Mode.TOOLS``) instead of JSON mode.
    """
    provider = provider.lower()

    if provider not in _PROVIDER_API_KEY_ENV:
        raise ProviderNotSupportedError(
            f"Unsupported provider: {provider}. Supported: {', '.join(_PROVIDER_API_KEY_ENV)}"
        )

    # Resolve API key — custom env var or provider default
    key_env = api_key_env or _PROVIDER_API_KEY_ENV[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise APIKeyError(
            f"{key_env} not set. Get your API key and set this environment variable."
        )

    os.environ[provider.upper() + "_API_KEY"] = api_key

    # Custom API base (for OpenAI-compatible endpoints)
    if api_base:
        os.environ[provider.upper() + "_API_BASE"] = api_base

    mode = instructor.Mode.TOOLS if model_supports_tools else instructor.Mode.JSON
    return instructor.from_litellm(completion, mode=mode)


def list_available_providers() -> dict[str, bool]:
    return {p: bool(os.getenv(e)) for p, e in _PROVIDER_API_KEY_ENV.items()}
