"""Unit tests for ChatMessage model."""

import pytest
from pydantic import ValidationError

from openai_client.exceptions.openai_validation_error import OpenAIValidationError
from openai_client.models.chat_message import ChatMessage


class TestChatMessage:
    """Test cases for ChatMessage model."""

    def test_valid_message(self):
        """Test creating a valid chat message."""
        message = ChatMessage(role="user", content="Hello, world!")

        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_message_with_name(self):
        """Test creating a message with a name."""
        message = ChatMessage(role="assistant", content="Hi there!", name="bot")

        assert message.role == "assistant"
        assert message.content == "Hi there!"
        assert message.name == "bot"

    def test_empty_content(self):
        """Test that empty content raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            ChatMessage(role="user", content="")

        assert "Message content cannot be empty" in str(exc_info.value)

    def test_whitespace_content(self):
        """Test that whitespace-only content raises OpenAIValidationError."""
        with pytest.raises(OpenAIValidationError) as exc_info:
            ChatMessage(role="user", content="   ")

        assert "Message content cannot be empty" in str(exc_info.value)

    def test_invalid_role(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="invalid", content="Hello")  # type: ignore[arg-type]

        assert "Input should be 'system', 'user' or 'assistant'" in str(exc_info.value)

    def test_model_dump(self):
        """Test model serialization."""
        message = ChatMessage(role="user", content="Test message")
        data = message.model_dump()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert data["name"] is None

    def test_model_dump_json(self):
        """Test model JSON serialization."""
        message = ChatMessage(role="user", content="Test message")
        json_data = message.model_dump_json()

        assert '"role":"user"' in json_data
        assert '"content":"Test message"' in json_data

    def test_hash(self):
        """Test that message is hashable."""
        message1 = ChatMessage(role="user", content="Hello")
        message2 = ChatMessage(role="user", content="Hello")
        message3 = ChatMessage(role="assistant", content="Hi")

        # Test that equal messages have same hash
        assert hash(message1) == hash(message2)

        # Test that different messages have different hashes
        assert hash(message1) != hash(message3)

        # Test that message can be used in set
        message_set = {message1, message2, message3}
        assert len(message_set) == 2  # message1 and message2 are equal

    def test_equality(self):
        """Test message equality."""
        message1 = ChatMessage(role="user", content="Hello")
        message2 = ChatMessage(role="user", content="Hello")
        message3 = ChatMessage(role="user", content="Hello", name="test")

        assert message1 == message2
        assert message1 != message3
        assert message1 != "not a message"

    def test_str_representation(self):
        """Test string representation."""
        message = ChatMessage(role="user", content="Hello, world!")
        str_repr = str(message)

        assert "user" in str_repr
        assert "Hello, world!" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        message = ChatMessage(role="assistant", content="Hi there!", name="bot")
        repr_str = repr(message)

        assert "ChatMessage" in repr_str
        assert "assistant" in repr_str
        assert "Hi there!" in repr_str
        assert "bot" in repr_str
