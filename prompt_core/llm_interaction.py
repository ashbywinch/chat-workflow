"""
Main LLM interaction module with multi-provider support using instructor and litellm.
"""

import os
from typing import Literal, Optional

from dotenv import load_dotenv

from .exceptions import (
    APIKeyError,
    ConfigurationError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
)

load_dotenv()

# Try to import instructor with litellm support
try:
    import instructor
    from litellm import completion
    import litellm

    # Suppress litellm's noisy "Provider List" debug output that fires in
    # the logging success handler for model names not in its registry
    # (e.g. openrouter/xiaomi/mimo-v2-flash). The errors are caught internally.
    litellm.suppress_debug_info = True

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    raise ProviderNotFoundError(
        "litellm not installed. Install with: uv add litellm\n"
        "This package is required for multi-provider LLM support."
    )

# Provider type for type hints
ProviderType = Literal[
    "openai", "google", "anthropic", "groq", "together", "azure", "openrouter"
]


def get_client(
    provider: Optional[ProviderType] = None,
    supports_tools: bool = False,
):
    """
    Get LLM client for the specified provider.

    Args:
        provider: One of "openai", "google", "anthropic", "groq", "together", "azure", "openrouter"
                 If None, uses provider from config
        supports_tools: Whether the model supports tool/function calling.
                       If True, uses Mode.TOOLS; otherwise Mode.JSON.

    Returns:
        Instructor-patched client for the specified provider

    Raises:
        ConfigurationError: If provider configuration is missing or invalid
        APIKeyError: If API key is missing for the provider
        ProviderNotSupportedError: If provider is not supported
        ProviderNotFoundError: If litellm is not installed
    """
    if not LITELLM_AVAILABLE:
        raise ProviderNotFoundError(
            "litellm not installed. Install with: uv add litellm\n"
            "This package is required for multi-provider LLM support."
        )

    # Use litellm for multi-provider support
    if provider is None:
        # Get provider from config, not environment or defaults
        from .config import config

        provider = config.provider

    if not provider:
        raise ConfigurationError("LLM provider not configured in config.json")

    provider = provider.lower()

    # Map provider to litellm model name and required API key
    provider_config = {
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "error_msg": "OPENAI_API_KEY not set. Get key from: https://platform.openai.com/api-keys",
        },
        "google": {
            "api_key_env": "GOOGLE_API_KEY",
            "error_msg": "GOOGLE_API_KEY not set. Get key from: https://makersuite.google.com/app/apikey",
        },
        "anthropic": {
            "api_key_env": "ANTHROPIC_API_KEY",
            "error_msg": "ANTHROPIC_API_KEY not set. Get key from: https://console.anthropic.com",
        },
        "groq": {
            "api_key_env": "GROQ_API_KEY",
            "error_msg": "GROQ_API_KEY not set. Get key from: https://console.groq.com",
        },
        "together": {
            "api_key_env": "TOGETHER_API_KEY",
            "error_msg": "TOGETHER_API_KEY not set. Get key from: https://together.ai",
        },
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "error_msg": "OPENROUTER_API_KEY not set. Get key from: https://openrouter.ai/keys",
        },
    }

    if provider not in provider_config:
        raise ProviderNotSupportedError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(provider_config.keys())}"
        )

    cfg = provider_config[provider]
    api_key = os.getenv(cfg["api_key_env"])

    if not api_key:
        raise APIKeyError(cfg["error_msg"])

    # Create litellm client with instructor patch
    # Use TOOLS mode when model supports function calling, otherwise JSON mode
    mode = instructor.Mode.TOOLS if supports_tools else instructor.Mode.JSON
    return instructor.from_litellm(completion, mode=mode)


def list_available_providers() -> dict:
    """
    List available LLM providers based on API keys in environment.

    Returns:
        Dictionary mapping provider names to availability status
    """
    providers = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "together": bool(os.getenv("TOGETHER_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
    }

    return providers
