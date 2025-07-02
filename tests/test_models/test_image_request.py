"""Unit tests for ImageRequest model."""

import pytest
from pydantic import ValidationError

from openai_client.exceptions.openai_validation_error import OpenAIValidationError
from openai_client.models.image_request import ImageRequest


class TestImageRequest:
    """Test cases for ImageRequest model."""

    def test_valid_request(self):
        """Test creating a valid image request."""
        request = ImageRequest(prompt="A beautiful sunset")

        assert request.prompt == "A beautiful sunset"
        assert request.n == 1
        assert request.size == "1024x1024"
        assert request.response_format == "url"
        assert request.user is None

    def test_request_with_all_parameters(self):
        """Test creating a request with all parameters."""
        request = ImageRequest(
            prompt="A cat in space",
            n=4,
            size="512x512",
            response_format="b64_json",
            user="test_user",
        )

        assert request.prompt == "A cat in space"
        assert request.n == 4
        assert request.size == "512x512"
        assert request.response_format == "b64_json"
        assert request.user == "test_user"

    def test_empty_prompt(self):
        """Test that empty prompt raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            ImageRequest(prompt="")

        assert "Image prompt cannot be empty" in str(exc_info.value)

    def test_whitespace_prompt(self):
        """Test that whitespace-only prompt raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            ImageRequest(prompt="   ")

        assert "Image prompt cannot be empty" in str(exc_info.value)

    def test_long_prompt(self):
        """Test that very long prompt raises OpenAIValidationError."""
        long_prompt = "A" * 1001
        with pytest.raises(OpenAIValidationError) as exc_info:
            ImageRequest(prompt=long_prompt)

        assert "Image prompt must be 1000 characters or less" in str(exc_info.value)

    def test_invalid_size(self):
        """Test that invalid size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ImageRequest(prompt="Test", size="invalid")  # type: ignore[arg-type]

        assert "size" in str(exc_info.value)

    def test_invalid_response_format(self):
        """Test that invalid response_format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ImageRequest(prompt="Test", response_format="invalid")  # type: ignore[arg-type]

        assert "response_format" in str(exc_info.value)

    def test_invalid_n(self):
        """Test that invalid n raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ImageRequest(prompt="Test", n=0)

        assert "n" in str(exc_info.value)

    def test_valid_sizes(self):
        """Test all valid size values."""
        valid_sizes = ["256x256", "512x512", "1024x1024"]

        for size in valid_sizes:
            request = ImageRequest(prompt="Test", size=size)  # type: ignore[arg-type]
            assert request.size == size

    def test_valid_response_formats(self):
        """Test all valid response format values."""
        valid_formats = ["url", "b64_json"]

        for fmt in valid_formats:
            request = ImageRequest(prompt="Test", response_format=fmt)  # type: ignore[arg-type]
            assert request.response_format == fmt

    def test_model_dump(self):
        """Test model serialization."""
        request = ImageRequest(prompt="Test prompt")
        data = request.model_dump()

        assert data["prompt"] == "Test prompt"
        assert data["n"] == 1
        assert data["size"] == "1024x1024"
        assert data["response_format"] == "url"
        assert data["user"] is None

    def test_model_dump_json(self):
        """Test model JSON serialization."""
        request = ImageRequest(prompt="Test prompt")
        json_data = request.model_dump_json()

        assert '"prompt":"Test prompt"' in json_data
        assert '"n":1' in json_data
        assert '"size":"1024x1024"' in json_data

    def test_equality(self):
        """Test request equality."""
        request1 = ImageRequest(prompt="Hello", size="512x512")
        request2 = ImageRequest(prompt="Hello", size="512x512")
        request3 = ImageRequest(prompt="World", size="512x512")

        assert request1 == request2
        assert request1 != request3
        assert request1 != "not a request"

    def test_str_representation(self):
        """Test string representation."""
        request = ImageRequest(prompt="Test prompt")
        str_repr = str(request)

        assert "Test prompt" in str_repr
        assert "1024x1024" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        request = ImageRequest(prompt="Test prompt", size="512x512")
        repr_str = repr(request)

        assert "ImageRequest" in repr_str
        assert "Test prompt" in repr_str
        assert "512x512" in repr_str
