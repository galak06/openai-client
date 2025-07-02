"""Configuration management for the OpenAI client."""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

from .exceptions import OpenAIConfigError


class OpenAIConfig(BaseModel):
    """
    Configuration class for OpenAI client settings.

    This class follows the Single Responsibility Principle by handling only
    configuration management. It uses Pydantic for validation and type safety.
    """

    # API Configuration
    api_key: str = Field(..., description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    organization: Optional[str] = Field(
        default=None, description="OpenAI organization ID"
    )

    # Request Configuration
    timeout: int = Field(
        default=60, ge=1, le=300, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed requests",
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")

    @validator("log_level")  # type: ignore[misc]
    def validate_log_level(cls: Any, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise OpenAIConfigError(
                f"Log level must be one of: {', '.join(valid_levels)}"
            )
        return v.upper()

    @validator("api_key")  # type: ignore[misc]
    def validate_api_key(cls: Any, v: str) -> str:
        """Validate that API key is not empty and has correct format."""
        if not v or not v.strip():
            raise OpenAIConfigError("API key cannot be empty")

        # Basic format validation for OpenAI API keys
        if not v.startswith("sk-"):
            raise OpenAIConfigError("API key must start with 'sk-'")

        if len(v) < 20:
            raise OpenAIConfigError("API key appears to be too short")

        return v.strip()

    @validator("base_url")  # type: ignore[misc]
    def validate_base_url(cls: Any, v: str) -> str:
        """Validate base URL format."""
        if not v.startswith(("http://", "https://")):
            raise OpenAIConfigError("Base URL must start with http:// or https://")
        return v.rstrip("/")

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "OpenAIConfig":
        """
        Create configuration from environment variables.

        Args:
            env_file: Optional path to .env file to load

        Returns:
            OpenAIConfig instance

        Raises:
            OpenAIConfigError: If required configuration is missing
        """
        # Load .env file if provided
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            # Try to load .env from current directory
            load_dotenv()

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIConfigError("OPENAI_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            log_level=os.getenv("OPENAI_LOG_LEVEL", "INFO"),
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "OpenAIConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            OpenAIConfig instance
        """
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return self.dict()  # type: ignore[no-any-return]

    def get_headers(self) -> dict[str, str]:
        """
        Get headers for API requests.

        Returns:
            Dictionary of headers including authorization
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        return headers
