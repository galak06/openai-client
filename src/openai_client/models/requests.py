"""Request models for OpenAI API calls."""

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

from ..core.exceptions import OpenAIValidationError


class ChatMessage(BaseModel):
    """
    A message in a chat conversation.

    This model represents a single message with role and content,
    following the OpenAI chat completion API format.
    """

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message author"
    )
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(
        default=None, description="The name of the message author"
    )

    @validator("content")  # type: ignore[misc]
    def validate_content(cls: Any, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise OpenAIValidationError("Message content cannot be empty")
        return v.strip()


class ChatRequest(BaseModel):
    """
    Request model for chat completion API.

    This model encapsulates all parameters needed for a chat completion
    request, with proper validation and defaults.
    """

    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    model: str = Field(..., description="The model to use for completion")

    # Optional parameters with sensible defaults
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Controls randomness in the response"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=4000, description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Controls diversity via nucleus sampling",
    )
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Penalty for using frequent tokens"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Penalty for using new tokens"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Stop sequences for generation"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")

    @validator("messages")  # type: ignore[misc]
    def validate_messages(cls: Any, v: List[ChatMessage]) -> List[ChatMessage]:
        """Validate that there is at least one message."""
        if not v:
            raise OpenAIValidationError("At least one message is required")
        return v

    @validator("model")  # type: ignore[misc]
    def validate_model(cls: Any, v: str) -> str:
        """Validate model name format."""
        if not v or not v.strip():
            raise OpenAIValidationError("Model name cannot be empty")
        return v.strip()


class EmbeddingRequest(BaseModel):
    """
    Request model for text embedding API.

    This model handles requests for generating embeddings from text input.
    """

    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(
        default="text-embedding-ada-002", description="The embedding model to use"
    )
    user: Optional[str] = Field(
        default=None, description="A unique identifier for the user"
    )

    @validator("input")  # type: ignore[misc]
    def validate_input(cls: Any, v: Union[str, List[str]]) -> Union[str, List[str]]:
        """Validate input is not empty."""
        if isinstance(v, str):
            if not v or not v.strip():
                raise OpenAIValidationError("Input text cannot be empty")
            return v.strip()
        elif isinstance(v, list):
            if not v:
                raise OpenAIValidationError("Input list cannot be empty")
            return [item.strip() if isinstance(item, str) else str(item) for item in v]
        else:
            raise OpenAIValidationError("Input must be a string or list of strings")


class ImageRequest(BaseModel):
    """
    Request model for image generation API.

    This model handles requests for generating images from text descriptions.
    """

    prompt: str = Field(..., description="Text description of the image to generate")
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: Literal["256x256", "512x512", "1024x1024"] = Field(
        default="1024x1024", description="Size of the generated image"
    )
    response_format: Literal["url", "b64_json"] = Field(
        default="url", description="Format of the response"
    )
    user: Optional[str] = Field(
        default=None, description="A unique identifier for the user"
    )

    @validator("prompt")  # type: ignore[misc]
    def validate_prompt(cls: Any, v: str) -> str:
        """Validate that prompt is not empty and within length limits."""
        if not v or not v.strip():
            raise OpenAIValidationError("Image prompt cannot be empty")

        if len(v) > 1000:
            raise OpenAIValidationError("Image prompt must be 1000 characters or less")

        return v.strip()


class ModelRequest(BaseModel):
    """
    Request model for listing available models.

    This model is used for the models list API endpoint.
    """

    # This is a simple request model as the models endpoint
    # typically doesn't require parameters
    pass
