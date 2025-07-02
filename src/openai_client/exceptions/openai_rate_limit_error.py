"""Exception raised when rate limit is exceeded."""

from .openai_api_error import OpenAIAPIError


class OpenAIRateLimitError(OpenAIAPIError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)
