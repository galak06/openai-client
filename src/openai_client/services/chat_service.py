"""Chat service for OpenAI chat completion operations."""

from typing import Any, Dict, List, Optional

from .base_service import BaseService


class ChatService(BaseService):
    """
    Service for handling chat completion operations.

    This service encapsulates all chat-related functionality and provides
    a clean interface for chat completions.
    """

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute chat completion.

        Args:
            **kwargs: Must include 'messages' and can include:
                - model: Model to use for completion
                - temperature: Controls randomness (0.0 to 2.0)
                - max_tokens: Maximum tokens to generate
                - stream: Whether to stream the response

        Returns:
            Chat completion response
        """
        messages = kwargs.get("messages")
        model = kwargs.get("model", "gpt-3.5-turbo")
        temperature = kwargs.get("temperature", 1.0)
        max_tokens = kwargs.get("max_tokens")
        stream = kwargs.get("stream", False)

        self.validate_required_params({"messages": messages}, ["messages"])

        if messages:
            self.log_operation(
                "chat_completion",
                model=model,
                message_count=len(messages),
                temperature=temperature,
            )

        if messages:
            return await self.client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
        else:
            raise ValueError("Messages cannot be None")

    async def simple_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        **kwargs: Any,
    ) -> str:
        """
        Simple chat completion with a single message.

        Args:
            message: User message
            system_prompt: Optional system prompt
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Assistant's response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        response = await self.execute(messages=messages, model=model, **kwargs)
        return response["choices"][0]["message"]["content"]

    async def conversation(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Multi-turn conversation completion.

        Args:
            messages: Conversation history
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Conversation response
        """
        return await self.execute(messages=messages, model=model, **kwargs)
