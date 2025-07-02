"""Validation utilities for the OpenAI client."""

import re
from typing import Any, Dict, List, Union

from ..exceptions import OpenAIValidationError


def validate_api_key(api_key: str) -> str:
    """
    Validate OpenAI API key format.

    Args:
        api_key: API key to validate

    Returns:
        Validated API key

    Raises:
        OpenAIValidationError: If API key is invalid
    """
    if not api_key or not api_key.strip():
        raise OpenAIValidationError("API key cannot be empty")

    if not api_key.startswith("sk-"):
        raise OpenAIValidationError("API key must start with 'sk-'")

    if len(api_key) < 20:
        raise OpenAIValidationError("API key appears to be too short")

    return api_key.strip()


def validate_model_name(model_name: str) -> str:
    """
    Validate OpenAI model name format.

    Args:
        model_name: Model name to validate

    Returns:
        Validated model name

    Raises:
        OpenAIValidationError: If model name is invalid
    """
    if not model_name or not model_name.strip():
        raise OpenAIValidationError("Model name cannot be empty")

    # Basic validation for common model patterns
    valid_patterns = [
        r"^gpt-\d+(\.\d+)?(-turbo)?$",  # GPT models
        r"^text-embedding-ada-\d+$",  # Embedding models
        r"^dall-e-\d+$",  # DALL-E models
        r"^whisper-\d+$",  # Whisper models
    ]

    model_name = model_name.strip()

    for pattern in valid_patterns:
        if re.match(pattern, model_name, re.IGNORECASE):
            return model_name

    # If no pattern matches, still allow it (for future models)
    return model_name


def validate_prompt_length(prompt: str, max_length: int = 4000) -> str:
    """
    Validate prompt length.

    Args:
        prompt: Prompt to validate
        max_length: Maximum allowed length

    Returns:
        Validated prompt

    Raises:
        OpenAIValidationError: If prompt is too long
    """
    if not prompt or not prompt.strip():
        raise OpenAIValidationError("Prompt cannot be empty")

    if len(prompt) > max_length:
        raise OpenAIValidationError(
            f"Prompt is too long ({len(prompt)} chars). Maximum allowed: {max_length}"
        )

    return prompt.strip()


def validate_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validate chat messages format.

    Args:
        messages: List of message dictionaries

    Returns:
        Validated messages

    Raises:
        OpenAIValidationError: If messages are invalid
    """
    if not messages:
        raise OpenAIValidationError("Messages list cannot be empty")

    valid_roles = {"system", "user", "assistant"}

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise OpenAIValidationError(f"Message {i} must be a dictionary")

        if "role" not in message:
            raise OpenAIValidationError(f"Message {i} missing 'role' field")

        if "content" not in message:
            raise OpenAIValidationError(f"Message {i} missing 'content' field")

        role = message["role"]
        if role not in valid_roles:
            raise OpenAIValidationError(
                f"Message {i} has invalid role '{role}'. Valid roles: {valid_roles}"
            )

        if not message["content"] or not message["content"].strip():
            raise OpenAIValidationError(f"Message {i} content cannot be empty")

    return messages


def validate_temperature(temperature: float) -> float:
    """
    Validate temperature parameter.

    Args:
        temperature: Temperature value to validate

    Returns:
        Validated temperature

    Raises:
        OpenAIValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise OpenAIValidationError("Temperature must be a number")

    if temperature < 0.0 or temperature > 2.0:
        raise OpenAIValidationError("Temperature must be between 0.0 and 2.0")

    return float(temperature)


def validate_max_tokens(max_tokens: int) -> int:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Max tokens value to validate

    Returns:
        Validated max tokens

    Raises:
        OpenAIValidationError: If max_tokens is invalid
    """
    if not isinstance(max_tokens, int):
        raise OpenAIValidationError("max_tokens must be an integer")

    if max_tokens < 1 or max_tokens > 4000:
        raise OpenAIValidationError("max_tokens must be between 1 and 4000")

    return max_tokens


def validate_image_size(size: str) -> str:
    """
    Validate image size parameter.

    Args:
        size: Image size to validate

    Returns:
        Validated image size

    Raises:
        OpenAIValidationError: If size is invalid
    """
    valid_sizes = {"256x256", "512x512", "1024x1024"}

    if size not in valid_sizes:
        raise OpenAIValidationError(
            f"Invalid image size '{size}'. Valid sizes: {valid_sizes}"
        )

    return size


def validate_response_format(response_format: str) -> str:
    """
    Validate response format parameter.

    Args:
        response_format: Response format to validate

    Returns:
        Validated response format

    Raises:
        OpenAIValidationError: If format is invalid
    """
    valid_formats = {"url", "b64_json"}

    if response_format not in valid_formats:
        raise OpenAIValidationError(
            f"Invalid response format '{response_format}'. Valid formats: {valid_formats}"
        )

    return response_format
