"""Exception raised when API quota is exceeded."""

from .openai_api_error import OpenAIAPIError


class OpenAIQuotaError(OpenAIAPIError):
    """Exception raised when API quota is exceeded."""

    def __init__(self, message: str = "API quota exceeded") -> None:
        super().__init__(message, status_code=429)
