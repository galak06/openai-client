"""Unit tests for ChatRequest model."""

import pytest
from pydantic import ValidationError

from openai_client.exceptions.openai_validation_error import OpenAIValidationError
from openai_client.models.chat_message import ChatMessage
from openai_client.models.chat_request import ChatRequest


class TestChatRequest:
    """Test cases for ChatRequest model."""

    def test_valid_request(self):
        """Test creating a valid chat request."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages, model="gpt-3.5-turbo")

        assert request.messages == messages
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 1.0
        assert request.max_tokens is None

    def test_request_with_all_parameters(self):
        """Test creating a request with all parameters."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="Hello"),
        ]
        request = ChatRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            stream=True,
        )

        assert request.messages == messages
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.top_p == 0.9
        assert request.frequency_penalty == 0.1
        assert request.presence_penalty == 0.1
        assert request.stop == ["END"]
        assert request.stream is True

    def test_empty_messages(self):
        """Test that empty messages raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            ChatRequest(messages=[], model="gpt-3.5-turbo")

        assert "At least one message is required" in str(exc_info.value)

    def test_none_messages(self):
        """Test that None messages raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(messages=None, model="gpt-3.5-turbo")  # type: ignore[arg-type]

        assert "messages" in str(exc_info.value)

    def test_empty_model(self):
        """Test that empty model raises OpenAIValidationError."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(OpenAIValidationError) as exc_info:
            ChatRequest(messages=messages, model="")

        assert "Model name cannot be empty" in str(exc_info.value)

    def test_whitespace_model(self):
        """Test that whitespace-only model raises OpenAIValidationError."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(OpenAIValidationError) as exc_info:
            ChatRequest(messages=messages, model="   ")

        assert "Model name cannot be empty" in str(exc_info.value)

    def test_invalid_temperature(self):
        """Test that invalid temperature raises ValidationError."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(messages=messages, model="gpt-3.5-turbo", temperature=2.1)

        assert "temperature" in str(exc_info.value)

    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens raises ValidationError."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(messages=messages, model="gpt-3.5-turbo", max_tokens=-1)

        assert "max_tokens" in str(exc_info.value)

    def test_model_dump(self):
        """Test model serialization."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages, model="gpt-3.5-turbo")
        data = request.model_dump()

        assert "messages" in data
        assert data["model"] == "gpt-3.5-turbo"
        assert data["temperature"] == 1.0

    def test_model_dump_json(self):
        """Test model JSON serialization."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages, model="gpt-3.5-turbo")
        json_data = request.model_dump_json()

        assert '"model":"gpt-3.5-turbo"' in json_data
        assert '"temperature":1.0' in json_data

    def test_equality(self):
        """Test request equality."""
        messages1 = [ChatMessage(role="user", content="Hello")]
        messages2 = [ChatMessage(role="user", content="Hello")]
        request1 = ChatRequest(messages=messages1, model="gpt-3.5-turbo")
        request2 = ChatRequest(messages=messages2, model="gpt-3.5-turbo")
        request3 = ChatRequest(messages=messages1, model="gpt-4")

        assert request1 == request2
        assert request1 != request3
        assert request1 != "not a request"

    def test_str_representation(self):
        """Test string representation."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages, model="gpt-3.5-turbo")
        str_repr = str(request)

        assert "gpt-3.5-turbo" in str_repr
        assert "Hello" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages, model="gpt-3.5-turbo")
        repr_str = repr(request)

        assert "ChatRequest" in repr_str
        assert "gpt-3.5-turbo" in repr_str
