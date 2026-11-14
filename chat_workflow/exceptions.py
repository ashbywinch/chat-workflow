"""
Custom exceptions for chat-workflow with helpful error messages.
"""


class ChatWorkflowError(Exception):
    """Base exception for all chat-workflow errors."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message

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


class ValidationError(ChatWorkflowError):
    """Validation errors for business rules."""

    pass


class AtomicWorkflowStepError(ChatWorkflowError):
    """Atomic workflow step errors."""

    pass


class TurnLimitExceededError(AtomicWorkflowStepError):
    """Maximum conversation turns exceeded."""

    def __init__(self, max_turns: int):
        message = f"Maximum conversation turns ({max_turns}) reached"
        super().__init__(message)


class AtomicWorkflowStepFailedError(AtomicWorkflowStepError):
    """LLM indicated the atomic workflow step should fail."""

    def __init__(self, reason: str):
        message = f"LLM indicated failure: {reason}"
        super().__init__(message)
        self.messages: list[dict[str, str]] | None = None


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
