"""Embedding service for OpenAI text embedding operations."""

from typing import Any, Dict, List, Union

from .base_service import BaseService


class EmbeddingService(BaseService):
    """
    Service for handling text embedding operations.

    This service encapsulates all embedding-related functionality and provides
    a clean interface for text embeddings.
    """

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute embedding creation.

        Args:
            **kwargs: Must include 'input_text' and can include:
                - model: Embedding model to use
                - user: User identifier

        Returns:
            Embedding response
        """
        input_text = kwargs.get("input_text")
        model = kwargs.get("model", "text-embedding-ada-002")
        user = kwargs.get("user")

        self.validate_required_params({"input_text": input_text}, ["input_text"])

        self.log_operation(
            "create_embedding",
            model=model,
            input_type=type(input_text).__name__,
        )

        if input_text:
            return await self.client.create_embedding(
                input_text=input_text,
                model=model,
                user=user,
            )
        else:
            raise ValueError("Input text cannot be None")

    async def embed_text(
        self, text: str, model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """
        Create embedding for a single text.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector
        """
        response = await self.execute(input_text=text, model=model)
        return response["data"][0]["embedding"]

    async def embed_texts(
        self, texts: List[str], model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of embedding vectors
        """
        response = await self.execute(input_text=texts, model=model)
        return [item["embedding"] for item in response["data"]]

    async def get_embedding_dimension(
        self, model: str = "text-embedding-ada-002"
    ) -> int:
        """
        Get the dimension of embeddings for a model.

        Args:
            model: Embedding model to use

        Returns:
            Embedding dimension
        """
        # Create a test embedding to get dimension
        test_text = "test"
        embedding = await self.embed_text(test_text, model)
        return len(embedding)
