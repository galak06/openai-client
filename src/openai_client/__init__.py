"""
OpenAI Client - A comprehensive, well-structured OpenAI client with advanced features.

This package provides a clean, type-safe interface to OpenAI's API with support for
chat completions, embeddings, image generation, and more.
"""

from .core.client import OpenAIClient
from .core.config import OpenAIConfig
from .core.exceptions import OpenAIAPIError, OpenAIError, OpenAIValidationError

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "OpenAIClient",
    "OpenAIConfig",
    "OpenAIError",
    "OpenAIAPIError",
    "OpenAIValidationError",
]
