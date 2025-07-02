"""Exception raised when input validation fails."""

from typing import Any, Optional

from .openai_error import OpenAIError


class OpenAIValidationError(OpenAIError):
    """Exception raised when input validation fails."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Optional[Any] = None
    ) -> None:
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value
