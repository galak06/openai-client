"""Embedding request model for OpenAI API."""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, validator

from ..exceptions import OpenAIValidationError


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
            # Check for empty or whitespace-only strings in the list
            for item in v:
                if isinstance(item, str) and (not item or not item.strip()):
                    raise OpenAIValidationError("Input text cannot be empty")
            return [item.strip() if isinstance(item, str) else str(item) for item in v]
        else:
            raise OpenAIValidationError("Input must be a string or list of strings")

    @validator("model")  # type: ignore[misc]
    def validate_model(cls: Any, v: str) -> str:
        """Validate model is not empty."""
        if not v or not v.strip():
            raise OpenAIValidationError("Model name cannot be empty")
        return v.strip()
