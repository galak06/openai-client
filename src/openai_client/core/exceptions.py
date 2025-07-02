"""Custom exceptions for the OpenAI client."""

from typing import Any, Dict, Optional


class OpenAIError(Exception):
    """Base exception for all OpenAI client errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OpenAIAPIError(OpenAIError):
    """Exception raised when the OpenAI API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        api_error: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, {"status_code": status_code, "api_error": api_error})
        self.status_code = status_code
        self.api_error = api_error


class OpenAIValidationError(OpenAIError):
    """Exception raised when input validation fails."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Optional[Any] = None
    ) -> None:
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class OpenAIConfigError(OpenAIError):
    """Exception raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        super().__init__(message, {"config_key": config_key})
        self.config_key = config_key


class OpenAIQuotaError(OpenAIAPIError):
    """Exception raised when API quota is exceeded."""

    def __init__(self, message: str = "API quota exceeded") -> None:
        super().__init__(message, status_code=429)


class OpenAIRateLimitError(OpenAIAPIError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)
