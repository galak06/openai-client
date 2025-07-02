"""Chat request model for OpenAI API."""

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

from ..exceptions import OpenAIValidationError
from .chat_message import ChatMessage


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
