"""Unit tests for ImageService class."""

from unittest.mock import AsyncMock, Mock

import pytest

from openai_client.exceptions import OpenAIError
from openai_client.services.image_service import ImageService


class TestImageService:
    """Test cases for ImageService class."""

    def test_init(self):
        """Test ImageService initialization."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ImageService(mock_client)

        assert service.client == mock_client
        assert service.logger == mock_client.logger

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execute method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(
            return_value={"data": [{"url": "https://example.com/image.png"}]}
        )

        service = ImageService(mock_client)

        result = await service.execute(prompt="A beautiful sunset")

        assert result == {"data": [{"url": "https://example.com/image.png"}]}
        mock_client.generate_image.assert_called_once_with(
            prompt="A beautiful sunset",
            n=1,
            size="1024x1024",
            response_format="url",
            user=None,
        )

    @pytest.mark.asyncio
    async def test_execute_with_all_parameters(self):
        """Test execute method with all parameters."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(
            return_value={"data": [{"url": "https://example.com/image.png"}]}
        )

        service = ImageService(mock_client)

        result = await service.execute(
            prompt="A cat in space", n=4, size="512x512", response_format="b64_json"
        )

        assert result == {"data": [{"url": "https://example.com/image.png"}]}
        mock_client.generate_image.assert_called_once_with(
            prompt="A cat in space",
            n=4,
            size="512x512",
            response_format="b64_json",
            user=None,
        )

    @pytest.mark.asyncio
    async def test_execute_missing_prompt(self):
        """Test execute method with missing prompt."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ImageService(mock_client)

        with pytest.raises(OpenAIError) as exc_info:
            await service.execute()

        assert "Missing required parameters: prompt" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_single_image_success(self):
        """Test successful generate_single_image method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(
            return_value={"data": [{"url": "https://example.com/image.png"}]}
        )

        service = ImageService(mock_client)

        result = await service.generate_single_image("A beautiful sunset")

        assert result == "https://example.com/image.png"
        mock_client.generate_image.assert_called_once_with(
            prompt="A beautiful sunset",
            n=1,
            size="1024x1024",
            response_format="url",
            user=None,
        )

    @pytest.mark.asyncio
    async def test_generate_multiple_images_success(self):
        """Test successful generate_multiple_images method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(
            return_value={
                "data": [
                    {"url": "https://example.com/image1.png"},
                    {"url": "https://example.com/image2.png"},
                ]
            }
        )

        service = ImageService(mock_client)

        result = await service.generate_multiple_images("A beautiful sunset", n=2)

        assert result == [
            "https://example.com/image1.png",
            "https://example.com/image2.png",
        ]
        mock_client.generate_image.assert_called_once_with(
            prompt="A beautiful sunset",
            n=2,
            size="1024x1024",
            response_format="url",
            user=None,
        )

    @pytest.mark.asyncio
    async def test_generate_single_image_with_custom_size(self):
        """Test generate_single_image method with custom size."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(
            return_value={"data": [{"url": "https://example.com/image.png"}]}
        )

        service = ImageService(mock_client)

        result = await service.generate_single_image(
            "A beautiful sunset", size="512x512"
        )

        assert result == "https://example.com/image.png"
        call_args = mock_client.generate_image.call_args
        assert call_args[1]["size"] == "512x512"

    @pytest.mark.asyncio
    async def test_generate_single_image_with_b64_format(self):
        """Test generate_single_image method with b64_json format."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(
            return_value={"data": [{"b64_json": "base64_encoded_image_data"}]}
        )

        service = ImageService(mock_client)

        result = await service.generate_single_image(
            "A beautiful sunset", response_format="b64_json"
        )

        assert result == "base64_encoded_image_data"
        call_args = mock_client.generate_image.call_args
        assert call_args[1]["response_format"] == "b64_json"

    @pytest.mark.asyncio
    async def test_log_operation_called(self):
        """Test that log_operation is called during execute."""
        mock_client = Mock()
        mock_logger = Mock()
        mock_client.logger = mock_logger
        mock_client.generate_image = AsyncMock(
            return_value={"data": [{"url": "https://example.com/image.png"}]}
        )

        service = ImageService(mock_client)

        await service.execute(prompt="A beautiful sunset")

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "generate_image" in log_call
        assert "prompt_length" in log_call

    @pytest.mark.asyncio
    async def test_client_error_propagation(self):
        """Test that client errors are propagated."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(side_effect=Exception("API Error"))

        service = ImageService(mock_client)

        with pytest.raises(Exception) as exc_info:
            await service.execute(prompt="A beautiful sunset")

        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_single_image_empty_response(self):
        """Test generate_single_image method with empty response."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(return_value={"data": []})

        service = ImageService(mock_client)

        with pytest.raises(IndexError) as exc_info:
            await service.generate_single_image("A beautiful sunset")

        assert "list index out of range" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_multiple_images_empty_response(self):
        """Test generate_multiple_images method with empty response."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(return_value={"data": []})

        service = ImageService(mock_client)

        result = await service.generate_multiple_images("A beautiful sunset")
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_single_image_missing_url(self):
        """Test generate_single_image method with missing url field."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.generate_image = AsyncMock(return_value={"data": [{}]})

        service = ImageService(mock_client)

        with pytest.raises(KeyError) as exc_info:
            await service.generate_single_image("A beautiful sunset")

        assert "url" in str(exc_info.value)
