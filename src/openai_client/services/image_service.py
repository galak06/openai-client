"""Image service for OpenAI image generation operations."""

from typing import Any, Dict, List

from .base_service import BaseService


class ImageService(BaseService):
    """
    Service for handling image generation operations.

    This service encapsulates all image-related functionality and provides
    a clean interface for image generation.
    """

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute image generation.

        Args:
            **kwargs: Must include 'prompt' and can include:
                - n: Number of images to generate
                - size: Image size
                - response_format: Response format (url or b64_json)
                - user: User identifier

        Returns:
            Image generation response
        """
        prompt = kwargs.get("prompt")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        response_format = kwargs.get("response_format", "url")
        user = kwargs.get("user")

        self.validate_required_params({"prompt": prompt}, ["prompt"])

        self.log_operation(
            "generate_image",
            prompt_length=len(prompt) if prompt else 0,
            n=n,
            size=size,
        )

        if prompt:
            return await self.client.generate_image(
                prompt=prompt,
                n=n,
                size=size,
                response_format=response_format,
                user=user,
            )
        else:
            raise ValueError("Prompt cannot be None")

    async def generate_single_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> str:
        """
        Generate a single image and return its URL or base64 data.

        Args:
            prompt: Text description of the image
            size: Image size
            response_format: Response format (url or b64_json)

        Returns:
            Image URL or base64 data
        """
        response = await self.execute(
            prompt=prompt,
            n=1,
            size=size,
            response_format=response_format,
        )
        return response["data"][0]["url" if response_format == "url" else "b64_json"]

    async def generate_multiple_images(
        self,
        prompt: str,
        n: int = 4,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> List[str]:
        """
        Generate multiple images and return their URLs or base64 data.

        Args:
            prompt: Text description of the image
            n: Number of images to generate (max 10)
            size: Image size
            response_format: Response format (url or b64_json)

        Returns:
            List of image URLs or base64 data
        """
        if n > 10:
            raise ValueError("Cannot generate more than 10 images at once")

        response = await self.execute(
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )

        key = "url" if response_format == "url" else "b64_json"
        return [item[key] for item in response["data"]]

    async def generate_image_variations(
        self,
        image_path: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> List[str]:
        """
        Generate image variations from an existing image.

        Args:
            image_path: Path to the source image
            n: Number of variations to generate
            size: Image size
            response_format: Response format (url or b64_json)

        Returns:
            List of variation URLs or base64 data
        """
        # Note: This would require implementing image variation endpoint
        # For now, this is a placeholder
        raise NotImplementedError("Image variations not yet implemented")
