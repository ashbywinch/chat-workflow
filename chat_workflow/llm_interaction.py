"""Multi-provider LLM client via instructor and litellm."""

import os
from typing import Literal

from dotenv import load_dotenv

from .exceptions import (
    APIKeyError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
)

load_dotenv()

try:
    import instructor
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

ProviderType = Literal[
    "openai", "google", "anthropic", "groq", "together", "azure", "openrouter"
]

_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_client(provider: str):
    """Get an instructor-patched LLM client for *provider*.

    The API key is read from the corresponding environment variable
    (e.g. ``OPENAI_API_KEY`` for ``provider="openai"``).
    """
    if not LITELLM_AVAILABLE:
        raise ProviderNotFoundError(
            "litellm not installed. Install with: uv add litellm"
        )

    provider = provider.lower()

    if provider not in _PROVIDER_API_KEY_ENV:
        raise ProviderNotSupportedError(
            f"Unsupported provider: {provider}. "
            f"Supported: {', '.join(_PROVIDER_API_KEY_ENV)}"
        )

    api_key = os.getenv(_PROVIDER_API_KEY_ENV[provider])
    if not api_key:
        raise APIKeyError(
            f"{_PROVIDER_API_KEY_ENV[provider]} not set. "
            f"Get your API key and set this environment variable."
        )

    os.environ[provider.upper() + "_API_KEY"] = api_key
    mode = instructor.Mode.JSON
    return instructor.from_litellm(completion, mode=mode)


def list_available_providers() -> dict[str, bool]:
    return {p: bool(os.getenv(e)) for p, e in _PROVIDER_API_KEY_ENV.items()}
