"""Image request model for OpenAI API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, validator

from ..exceptions import OpenAIValidationError


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
