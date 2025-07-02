"""Exception raised when the OpenAI API returns an error."""

from typing import Any, Dict, Optional

from .openai_error import OpenAIError


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
