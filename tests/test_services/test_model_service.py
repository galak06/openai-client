"""Unit tests for ModelService class."""

from unittest.mock import AsyncMock, Mock

import pytest

from openai_client.exceptions import OpenAIError
from openai_client.services.model_service import ModelService


class TestModelService:
    """Test cases for ModelService class."""

    def test_init(self):
        """Test ModelService initialization."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ModelService(mock_client)

        assert service.client == mock_client
        assert service.logger == mock_client.logger

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execute method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.list_models = AsyncMock(
            return_value={
                "data": [
                    {"id": "gpt-3.5-turbo", "object": "model"},
                    {"id": "gpt-4", "object": "model"},
                ]
            }
        )

        service = ModelService(mock_client)

        result = await service.execute()

        assert result == {
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4", "object": "model"},
            ]
        }
        mock_client.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test successful list_models method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.list_models = AsyncMock(
            return_value={
                "data": [
                    {"id": "gpt-3.5-turbo", "object": "model"},
                    {"id": "gpt-4", "object": "model"},
                ]
            }
        )

        service = ModelService(mock_client)

        result = await service.list_models()

        assert result == [
            {"id": "gpt-3.5-turbo", "object": "model"},
            {"id": "gpt-4", "object": "model"},
        ]
        mock_client.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_info_success(self):
        """Test successful get_model_info method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.get_model = AsyncMock(
            return_value={
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
            }
        )

        service = ModelService(mock_client)

        result = await service.get_model_info("gpt-3.5-turbo")

        assert result == {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
        }
        mock_client.get_model.assert_called_once_with("gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_get_chat_models_success(self):
        """Test successful get_chat_models method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.list_models = AsyncMock(
            return_value={
                "data": [
                    {"id": "gpt-3.5-turbo", "object": "model"},
                    {"id": "gpt-4", "object": "model"},
                    {"id": "text-embedding-ada-002", "object": "model"},
                    {"id": "dall-e-3", "object": "model"},
                ]
            }
        )

        service = ModelService(mock_client)

        result = await service.get_chat_models()

        assert len(result) == 2
        assert any(model["id"] == "gpt-3.5-turbo" for model in result)
        assert any(model["id"] == "gpt-4" for model in result)
        assert not any(model["id"] == "text-embedding-ada-002" for model in result)
        assert not any(model["id"] == "dall-e-3" for model in result)

    @pytest.mark.asyncio
    async def test_get_embedding_models_success(self):
        """Test successful get_embedding_models method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.list_models = AsyncMock(
            return_value={
                "data": [
                    {"id": "gpt-3.5-turbo", "object": "model"},
                    {"id": "text-embedding-ada-002", "object": "model"},
                    {"id": "text-embedding-3-small", "object": "model"},
                ]
            }
        )

        service = ModelService(mock_client)

        result = await service.get_embedding_models()

        assert len(result) == 2
        assert any(model["id"] == "text-embedding-ada-002" for model in result)
        assert any(model["id"] == "text-embedding-3-small" for model in result)
        assert not any(model["id"] == "gpt-3.5-turbo" for model in result)

    @pytest.mark.asyncio
    async def test_log_operation_called(self):
        """Test that log_operation is called during execute."""
        mock_client = Mock()
        mock_logger = Mock()
        mock_client.logger = mock_logger
        mock_client.list_models = AsyncMock(
            return_value={"data": [{"id": "gpt-3.5-turbo", "object": "model"}]}
        )

        service = ModelService(mock_client)

        await service.execute()

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "model_operation" in log_call
        assert "operation_type" in log_call

    @pytest.mark.asyncio
    async def test_client_error_propagation(self):
        """Test that client errors are propagated."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.list_models = AsyncMock(side_effect=Exception("API Error"))

        service = ModelService(mock_client)

        with pytest.raises(Exception) as exc_info:
            await service.execute()

        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_models_empty_response(self):
        """Test list_models method with empty response."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.list_models = AsyncMock(return_value={"data": []})

        service = ModelService(mock_client)

        result = await service.list_models()
        assert result == []
