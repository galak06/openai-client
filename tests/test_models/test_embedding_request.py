"""Unit tests for EmbeddingRequest model."""

import pytest
from pydantic import ValidationError

from openai_client.exceptions.openai_validation_error import OpenAIValidationError
from openai_client.models.embedding_request import EmbeddingRequest


class TestEmbeddingRequest:
    """Test cases for EmbeddingRequest model."""

    def test_valid_single_text(self):
        """Test creating a valid embedding request with single text."""
        request = EmbeddingRequest(input="Hello, world!")

        assert request.input == "Hello, world!"
        assert request.model == "text-embedding-ada-002"
        assert request.user is None

    def test_valid_multiple_texts(self):
        """Test creating a valid embedding request with multiple texts."""
        texts = ["Hello", "World", "Test"]
        request = EmbeddingRequest(input=texts)

        assert request.input == texts
        assert request.model == "text-embedding-ada-002"
        assert request.user is None

    def test_request_with_custom_model(self):
        """Test creating a request with custom model."""
        request = EmbeddingRequest(
            input="Test text", model="text-embedding-3-small", user="test_user"
        )

        assert request.input == "Test text"
        assert request.model == "text-embedding-3-small"
        assert request.user == "test_user"

    def test_empty_input(self):
        """Test that empty input raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            EmbeddingRequest(input="")

        assert "Input text cannot be empty" in str(exc_info.value)

    def test_whitespace_input(self):
        """Test that whitespace-only input raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            EmbeddingRequest(input="   ")

        assert "Input text cannot be empty" in str(exc_info.value)

    def test_empty_list_input(self):
        """Test that empty list input raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            EmbeddingRequest(input=[])

        assert "Input list cannot be empty" in str(exc_info.value)

    def test_list_with_empty_strings(self):
        """Test that list with empty strings raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            EmbeddingRequest(input=["", "Hello", ""])

        assert "Input text cannot be empty" in str(exc_info.value)

    def test_list_with_whitespace_strings(self):
        """Test that list with whitespace strings raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            EmbeddingRequest(input=["   ", "Hello", "   "])

        assert "Input text cannot be empty" in str(exc_info.value)

    def test_invalid_model(self):
        """Test that invalid model raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            EmbeddingRequest(input="Test", model="")  # type: ignore[arg-type]

        assert "Model name cannot be empty" in str(exc_info.value)

    def test_model_dump(self):
        """Test model serialization."""
        request = EmbeddingRequest(input="Test text", model="text-embedding-ada-002")
        data = request.model_dump()

        assert data["input"] == "Test text"
        assert data["model"] == "text-embedding-ada-002"
        assert data["user"] is None

    def test_model_dump_json(self):
        """Test model JSON serialization."""
        request = EmbeddingRequest(input="Test text", model="text-embedding-ada-002")
        json_data = request.model_dump_json()

        assert '"input":"Test text"' in json_data
        assert '"model":"text-embedding-ada-002"' in json_data

    def test_equality(self):
        """Test request equality."""
        request1 = EmbeddingRequest(input="Hello", model="text-embedding-ada-002")
        request2 = EmbeddingRequest(input="Hello", model="text-embedding-ada-002")
        request3 = EmbeddingRequest(input="World", model="text-embedding-ada-002")

        assert request1 == request2
        assert request1 != request3
        assert request1 != "not a request"

    def test_str_representation(self):
        """Test string representation."""
        request = EmbeddingRequest(input="Test text")
        str_repr = str(request)

        assert "Test text" in str_repr
        assert "text-embedding-ada-002" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        request = EmbeddingRequest(input="Test text", model="text-embedding-ada-002")
        repr_str = repr(request)

        assert "EmbeddingRequest" in repr_str
        assert "Test text" in repr_str
        assert "text-embedding-ada-002" in repr_str

    def test_user_parameter(self):
        """Test user parameter."""
        request = EmbeddingRequest(input="Test", user="test_user")
        assert request.user == "test_user"

        request = EmbeddingRequest(input="Test", user=None)
        assert request.user is None

    def test_long_text(self):
        """Test that very long text is accepted."""
        long_text = "A" * 10000
        request = EmbeddingRequest(input=long_text)
        assert request.input == long_text

    def test_special_characters(self):
        """Test text with special characters."""
        special_text = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"
        request = EmbeddingRequest(input=special_text)
        assert request.input == special_text

    def test_unicode_text(self):
        """Test text with unicode characters."""
        unicode_text = "Hello 世界! 🌍"
        request = EmbeddingRequest(input=unicode_text)
        assert request.input == unicode_text

    def test_valid_models(self):
        """Test all valid model names."""
        valid_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

        for model in valid_models:
            request = EmbeddingRequest(input="Test", model=model)
            assert request.model == model

    def test_none_input(self):
        """Test that None input raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingRequest(input=None)  # type: ignore[arg-type]

        assert "input" in str(exc_info.value)

    def test_repr(self):
        """Test request repr representation."""
        request = EmbeddingRequest(input="Test text", model="text-embedding-ada-002")
        repr_str = repr(request)

        assert "EmbeddingRequest" in repr_str
        assert "Test text" in repr_str
        assert "text-embedding-ada-002" in repr_str

    def test_str(self):
        """Test request string conversion."""
        request = EmbeddingRequest(input="Test text")
        str_repr = str(request)

        assert "Test text" in str_repr
        assert "text-embedding-ada-002" in str_repr
