"""Model service for OpenAI model information operations."""

from typing import Any, Dict, List

from .base_service import BaseService


class ModelService(BaseService):
    """
    Service for handling model information operations.

    This service encapsulates all model-related functionality and provides
    a clean interface for model information.
    """

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute model operation.

        Args:
            **kwargs: Can include:
                - model_id: Specific model ID to get info for
                - operation: 'list' or 'get'

        Returns:
            Model information response
        """
        operation = kwargs.get("operation", "list")
        model_id = kwargs.get("model_id")

        self.log_operation(
            "model_operation", operation_type=operation, model_id=model_id
        )

        if operation == "list":
            return await self.client.list_models()
        elif operation == "get":
            if not model_id:
                raise ValueError("model_id is required for 'get' operation")
            return await self.client.get_model(model_id)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all available models.

        Returns:
            List of model information dictionaries
        """
        response = await self.execute(operation="list")
        return response["data"]

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model information dictionary
        """
        response = await self.execute(operation="get", model_id=model_id)
        return response

    async def get_chat_models(self) -> List[Dict[str, Any]]:
        """
        Get list of models that support chat completion.

        Returns:
            List of chat model information
        """
        models = await self.list_models()
        return [model for model in models if "gpt" in model["id"].lower()]

    async def get_embedding_models(self) -> List[Dict[str, Any]]:
        """
        Get list of models that support embeddings.

        Returns:
            List of embedding model information
        """
        models = await self.list_models()
        return [model for model in models if "embedding" in model["id"].lower()]

    async def model_exists(self, model_id: str) -> bool:
        """
        Check if a model exists.

        Args:
            model_id: Model identifier

        Returns:
            True if model exists, False otherwise
        """
        try:
            await self.get_model_info(model_id)
            return True
        except Exception:
            return False

    async def get_model_capabilities(self, model_id: str) -> List[str]:
        """
        Get capabilities of a specific model.

        Args:
            model_id: Model identifier

        Returns:
            List of model capabilities
        """
        model_info = await self.get_model_info(model_id)
        capabilities = []

        # Add capabilities based on model ID patterns
        if "gpt" in model_id.lower():
            capabilities.append("chat_completion")
        if "embedding" in model_id.lower():
            capabilities.append("embeddings")
        if "dall-e" in model_id.lower():
            capabilities.append("image_generation")

        return capabilities
