"""Unit tests for EmbeddingService class."""

from unittest.mock import AsyncMock, Mock

import pytest

from openai_client.exceptions import OpenAIError
from openai_client.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService class."""

    def test_init(self):
        """Test EmbeddingService initialization."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = EmbeddingService(mock_client)

        assert service.client == mock_client
        assert service.logger == mock_client.logger

    @pytest.mark.asyncio
    async def test_execute_single_text_success(self):
        """Test successful execute method with single text."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        service = EmbeddingService(mock_client)

        result = await service.execute(input_text="Hello, world!")

        assert result == {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client.create_embedding.assert_called_once_with(
            input_text="Hello, world!", model="text-embedding-ada-002", user=None
        )

    @pytest.mark.asyncio
    async def test_execute_multiple_texts_success(self):
        """Test successful execute method with multiple texts."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
            }
        )

        service = EmbeddingService(mock_client)

        texts = ["Hello", "World"]
        result = await service.execute(input_text=texts)

        assert result == {
            "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
        }
        mock_client.create_embedding.assert_called_once_with(
            input_text=texts, model="text-embedding-ada-002", user=None
        )

    @pytest.mark.asyncio
    async def test_execute_with_all_parameters(self):
        """Test execute method with all parameters."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        service = EmbeddingService(mock_client)

        result = await service.execute(
            input_text="Hello, world!", model="text-embedding-3-small", user="test_user"
        )

        assert result == {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client.create_embedding.assert_called_once_with(
            input_text="Hello, world!", model="text-embedding-3-small", user="test_user"
        )

    @pytest.mark.asyncio
    async def test_execute_missing_input(self):
        """Test execute method with missing input."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = EmbeddingService(mock_client)

        with pytest.raises(OpenAIError) as exc_info:
            await service.execute()

        assert "Missing required parameters: input_text" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        """Test successful embed_text method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        service = EmbeddingService(mock_client)

        result = await service.embed_text("Hello, world!")

        assert result == [0.1, 0.2, 0.3]
        mock_client.create_embedding.assert_called_once_with(
            input_text="Hello, world!", model="text-embedding-ada-002", user=None
        )

    @pytest.mark.asyncio
    async def test_embed_texts_success(self):
        """Test successful embed_texts method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
            }
        )

        service = EmbeddingService(mock_client)

        texts = ["Hello", "World"]
        result = await service.embed_texts(texts)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.create_embedding.assert_called_once_with(
            input_text=texts, model="text-embedding-ada-002", user=None
        )

    @pytest.mark.asyncio
    async def test_embed_text_with_custom_model(self):
        """Test embed_text method with custom model."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        service = EmbeddingService(mock_client)

        result = await service.embed_text(
            "Hello, world!", model="text-embedding-3-small"
        )

        assert result == [0.1, 0.2, 0.3]
        call_args = mock_client.create_embedding.call_args
        assert call_args[1]["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_texts_with_custom_model(self):
        """Test embed_texts method with custom model."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
            }
        )

        service = EmbeddingService(mock_client)

        texts = ["Hello", "World"]
        result = await service.embed_texts(texts, model="text-embedding-3-small")

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        call_args = mock_client.create_embedding.call_args
        assert call_args[1]["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_text_with_user(self):
        """Test embed_text method with user parameter."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        service = EmbeddingService(mock_client)

        # Note: embed_text method doesn't support user parameter directly
        result = await service.embed_text("Hello, world!")

        assert result == [0.1, 0.2, 0.3]
        call_args = mock_client.create_embedding.call_args
        assert call_args[1]["model"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_embed_texts_with_user(self):
        """Test embed_texts method with user parameter."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
            }
        )

        service = EmbeddingService(mock_client)

        texts = ["Hello", "World"]
        result = await service.embed_texts(texts)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        call_args = mock_client.create_embedding.call_args
        assert call_args[1]["model"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_log_operation_called(self):
        """Test that log_operation is called during execute."""
        mock_client = Mock()
        mock_logger = Mock()
        mock_client.logger = mock_logger
        mock_client.create_embedding = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        service = EmbeddingService(mock_client)

        await service.execute(input_text="Hello, world!")

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "create_embedding" in log_call
        assert "text-embedding-ada-002" in log_call
        assert "str" in log_call  # input_type

    @pytest.mark.asyncio
    async def test_client_error_propagation(self):
        """Test that client errors are propagated."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(side_effect=Exception("API Error"))

        service = EmbeddingService(mock_client)

        with pytest.raises(Exception) as exc_info:
            await service.execute(input_text="Hello, world!")

        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_text_empty_response(self):
        """Test embed_text method with empty response."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(return_value={"data": []})

        service = EmbeddingService(mock_client)

        with pytest.raises(IndexError) as exc_info:
            await service.embed_text("Hello, world!")

        assert "list index out of range" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_texts_empty_response(self):
        """Test embed_texts method with empty response."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(return_value={"data": []})

        service = EmbeddingService(mock_client)

        result = await service.embed_texts(["Hello", "World"])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_text_missing_embedding(self):
        """Test embed_text method with missing embedding field."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.create_embedding = AsyncMock(return_value={"data": [{}]})

        service = EmbeddingService(mock_client)

        with pytest.raises(KeyError) as exc_info:
            await service.embed_text("Hello, world!")

        assert "embedding" in str(exc_info.value)
