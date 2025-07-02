"""Base exception for all OpenAI client errors."""

from typing import Any, Dict, Optional


class OpenAIError(Exception):
    """Base exception for all OpenAI client errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
