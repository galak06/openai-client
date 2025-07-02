"""Exception raised when configuration is invalid or missing."""

from typing import Optional

from .openai_error import OpenAIError


class OpenAIConfigError(OpenAIError):
    """Exception raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        super().__init__(message, {"config_key": config_key})
        self.config_key = config_key
