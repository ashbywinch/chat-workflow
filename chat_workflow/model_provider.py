"""How the configured provider becomes the one model the library calls.

The library's provider names are a product promise; the objects that talk to
them are PydanticAI's, and they differ per service. This module is the only
place that knows either fact, so a provider is added or narrowed in one file
and no workflow, step or test has to learn an SDK.

Only OpenAI-compatible services are wired in the ``[openai]`` build: the house
gateway, OpenRouter, Groq, Together and Azure. Google and Anthropic need their
own SDK extra and are not installed; asking for them says so rather than
failing later with an import error from inside the agent loop.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

from .exceptions import APIKeyError, ProviderNotSupportedError

# Reads a local .env for development, exactly as the old client did. Real
# environment variables win; a missing .env is not an error.
load_dotenv()

ProviderType = Literal["openai", "google", "anthropic", "groq", "together", "azure", "openrouter"]

PROVIDERS: frozenset[str] = frozenset(
    ("openai", "google", "anthropic", "groq", "together", "azure", "openrouter")
)

_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Services that speak the OpenAI wire format but have their own host. The
# gateway and a direct OpenAI account have no default here: their endpoint is
# configuration (OPENAI_BASE_URL or api_base), never a guess.
_OPENAI_COMPATIBLE_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}

# The SDK extra each unwired service needs. Naming it is the whole fix; an
# "unsupported provider" with no next step is a dead end.
_EXTRA_FOR = {
    "google": "google",
    "anthropic": "anthropic",
}


def build_model(
    *,
    provider: str,
    model: str,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> Model:
    """The model *provider*/*model* names, ready for a conversation.

    The key comes from ``api_key_env`` when the caller names one, else the
    provider's own environment variable. ``api_base`` overrides the service's
    host (the gateway is such an override); an OpenAI-compatible service with
    no override uses its published host.
    """
    provider = provider.lower()
    if provider in _EXTRA_FOR:
        raise ProviderNotSupportedError(
            f"Provider {provider!r} needs its own SDK extra, which this build does not install. "
            f'Install it with `pip install "pydantic-ai-slim[{_EXTRA_FOR[provider]}]"`, '
            "or use the house gateway (provider 'openai')."
        )
    if provider not in PROVIDERS:
        raise ProviderNotSupportedError(
            f"Unsupported provider: {provider}. Supported: {', '.join(sorted(PROVIDERS))}"
        )

    key_env = api_key_env or _KEY_ENV[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise APIKeyError(f"{key_env} not set. Get your API key and set this environment variable.")

    if provider == "azure":
        # Azure needs its resource endpoint and, unless it is the v1 API, an
        # api-version. The endpoint is the configured api_base, falling back to
        # Azure's own environment variable; PydanticAI reads the version from
        # the environment.
        return OpenAIChatModel(
            model,
            provider=AzureProvider(
                azure_endpoint=api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key,
            ),
        )
    if provider == "openai":
        return OpenAIChatModel(model, provider=OpenAIProvider(base_url=api_base, api_key=api_key))
    return OpenAIChatModel(
        model,
        provider=OpenAICompatibleProvider(
            provider,
            base_url=api_base or _OPENAI_COMPATIBLE_BASE_URLS[provider],
            api_key=api_key,
        ),
    )


def list_available_providers() -> dict[str, bool]:
    """Which providers have a key in the environment, for CLI/help output."""
    return {provider: bool(os.getenv(env)) for provider, env in _KEY_ENV.items()}
