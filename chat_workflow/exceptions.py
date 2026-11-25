"""
Custom exceptions for chat-workflow with helpful error messages.
"""

from typing import Any, override

# lucidlint: ignore-file class-module the exceptions are one hierarchy; one module keeps the family together
__all__ = [
    "APIError",
    "APIKeyError",
    "AtomicWorkflowError",
    "AtomicWorkflowFailedError",
    "AuthenticationError",
    "ChatWorkflowError",
    "ConfigFileError",
    "ConfigurationError",
    "ConnectionError",
    "InvalidResponseError",
    "MaxRetriesExceededError",
    "ModelError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderNotSupportedError",
    "RateLimitError",
    "TurnLimitExceededError",
    "ValidationError",
]


class ChatWorkflowError(Exception):
    """Base exception for all chat-workflow errors."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message

    @override
    def __str__(self):
        return self.message if self.message else super().__str__()


class ConfigurationError(ChatWorkflowError):
    """Configuration-related errors."""

    pass


class ConfigFileError(ConfigurationError):
    """Errors related to config.json file."""

    pass


class APIKeyError(ConfigurationError):
    """Missing or invalid API key."""

    pass


class ProviderError(ChatWorkflowError):
    """LLM provider-related errors."""

    pass


class ProviderNotSupportedError(ProviderError):
    """Requested provider is not supported."""

    pass


class ProviderNotFoundError(ProviderError):
    """Provider module not found."""

    pass


class ValidationError(ChatWorkflowError, ValueError):
    """An answer the caller's business rules reject.

    It is a ``ValueError`` because that is the only way a rule stated on an
    answer model reaches the framework's output-retry channel: Pydantic turns
    a ``ValueError`` from a validator into a ``pydantic.ValidationError``,
    whose sentence is sent back to the model by the same retry that handles a
    missing or mistyped field. Raised from a validator without this, the
    exception escapes validation and a model that forgot one rule crashes the
    workflow instead of being told.
    """

    pass


class AtomicWorkflowError(ChatWorkflowError):
    """Atomic workflow errors."""

    pass


class TurnLimitExceededError(AtomicWorkflowError):
    """Maximum conversation turns exceeded."""

    def __init__(self, max_turns: int):
        message = f"Maximum conversation turns ({max_turns}) reached"
        super().__init__(message)


class AtomicWorkflowFailedError(AtomicWorkflowError):
    """LLM indicated the atomic workflow should fail."""

    def __init__(self, reason: str):
        message = f"LLM indicated failure: {reason}"
        super().__init__(message)
        self.messages: list[Any] | None = None


class APIError(ChatWorkflowError):
    """External API errors."""

    pass


class AuthenticationError(APIError):
    """API authentication failed."""

    pass


class ConnectionError(APIError):
    """Network connection failed."""

    pass


class RateLimitError(APIError):
    """API rate limit exceeded."""

    pass


class ModelError(ChatWorkflowError):
    """Model-related errors."""

    pass


class InvalidResponseError(ModelError):
    """LLM returned invalid response format."""

    pass


class MaxRetriesExceededError(ModelError):
    """Maximum retries exceeded for LLM call."""

    def __init__(self, max_retries: int, reason: str = ""):
        message = f"Maximum retries ({max_retries}) exceeded for LLM call"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)
        self.reason = reason
