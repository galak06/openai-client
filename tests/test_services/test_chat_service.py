"""Unit tests for ChatService class."""

from unittest.mock import AsyncMock, Mock

import pytest

from openai_client.exceptions import OpenAIError
from openai_client.services.chat_service import ChatService


class TestChatService:
    """Test cases for ChatService class."""

    def test_init(self):
        """Test ChatService initialization."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ChatService(mock_client)

        assert service.client == mock_client
        assert service.logger == mock_client.logger

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execute method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello"}}]}
        )

        service = ChatService(mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        result = await service.execute(messages=messages)

        assert result == {"choices": [{"message": {"content": "Hello"}}]}
        mock_client.chat_completion.assert_called_once_with(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=1.0,
            max_tokens=None,
            stream=False,
        )

    @pytest.mark.asyncio
    async def test_execute_with_all_parameters(self):
        """Test execute method with all parameters."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello"}}]}
        )

        service = ChatService(mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        result = await service.execute(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        assert result == {"choices": [{"message": {"content": "Hello"}}]}
        mock_client.chat_completion.assert_called_once_with(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

    @pytest.mark.asyncio
    async def test_execute_missing_messages(self):
        """Test execute method with missing messages."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ChatService(mock_client)

        with pytest.raises(OpenAIError) as exc_info:
            await service.execute()

        assert "Missing required parameters: messages" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_none_messages(self):
        """Test execute method with None messages."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ChatService(mock_client)

        with pytest.raises(OpenAIError) as exc_info:
            await service.execute(messages=None)

        assert "Missing required parameters: messages" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_empty_messages(self):
        """Test execute method with empty messages."""
        mock_client = Mock()
        mock_client.logger = Mock()

        service = ChatService(mock_client)

        with pytest.raises(ValueError) as exc_info:
            await service.execute(messages=[])

        assert "Messages cannot be None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_simple_chat_success(self):
        """Test successful simple_chat method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        result = await service.simple_chat("Hello")

        assert result == "Hello there!"
        mock_client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_simple_chat_with_system_prompt(self):
        """Test simple_chat method with system prompt."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        result = await service.simple_chat("Hello", system_prompt="You are helpful")

        assert result == "Hello there!"
        mock_client.chat_completion.assert_called_once()

        # Check that the messages include system prompt
        call_args = mock_client.chat_completion.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_simple_chat_with_custom_model(self):
        """Test simple_chat method with custom model."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        result = await service.simple_chat("Hello", model="gpt-4")

        assert result == "Hello there!"
        call_args = mock_client.chat_completion.call_args
        assert call_args[1]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_simple_chat_with_additional_kwargs(self):
        """Test simple_chat method with additional kwargs."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        result = await service.simple_chat("Hello", temperature=0.7, max_tokens=100)

        assert result == "Hello there!"
        call_args = mock_client.chat_completion.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_conversation_success(self):
        """Test successful conversation method."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = await service.conversation(messages)

        assert result == {"choices": [{"message": {"content": "Hello there!"}}]}
        mock_client.chat_completion.assert_called_once_with(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=1.0,
            max_tokens=None,
            stream=False,
        )

    @pytest.mark.asyncio
    async def test_conversation_with_custom_model(self):
        """Test conversation method with custom model."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        result = await service.conversation(messages, model="gpt-4")

        assert result == {"choices": [{"message": {"content": "Hello there!"}}]}
        call_args = mock_client.chat_completion.call_args
        assert call_args[1]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_conversation_with_additional_kwargs(self):
        """Test conversation method with additional kwargs."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello there!"}}]}
        )

        service = ChatService(mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        result = await service.conversation(messages, temperature=0.7, max_tokens=100)

        assert result == {"choices": [{"message": {"content": "Hello there!"}}]}
        call_args = mock_client.chat_completion.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_log_operation_called(self):
        """Test that log_operation is called during execute."""
        mock_client = Mock()
        mock_logger = Mock()
        mock_client.logger = mock_logger
        mock_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello"}}]}
        )

        service = ChatService(mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        await service.execute(messages=messages)

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "chat_completion" in log_call
        assert "gpt-3.5-turbo" in log_call
        assert "1" in log_call  # message_count
        assert "1.0" in log_call  # temperature

    @pytest.mark.asyncio
    async def test_client_error_propagation(self):
        """Test that client errors are propagated."""
        mock_client = Mock()
        mock_client.logger = Mock()
        mock_client.chat_completion = AsyncMock(side_effect=Exception("API Error"))

        service = ChatService(mock_client)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception) as exc_info:
            await service.execute(messages=messages)

        assert "API Error" in str(exc_info.value)
