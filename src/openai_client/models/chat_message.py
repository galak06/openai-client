"""Chat message model for OpenAI API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, validator

from ..exceptions import OpenAIValidationError


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

    def __hash__(self) -> int:
        """Make the message hashable."""
        return hash((self.role, self.content, self.name))

    def __eq__(self, other: Any) -> bool:
        """Compare messages for equality."""
        if not isinstance(other, ChatMessage):
            return False
        return (
            self.role == other.role
            and self.content == other.content
            and self.name == other.name
        )
