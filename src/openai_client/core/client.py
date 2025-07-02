"""Main OpenAI client for handling API requests."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Union

import httpx
from openai import AsyncOpenAI, OpenAI

from ..exceptions import (
    OpenAIAPIError,
    OpenAIError,
    OpenAIQuotaError,
    OpenAIRateLimitError,
    OpenAIValidationError,
)
from ..utils.validators import (
    validate_image_size,
    validate_max_tokens,
    validate_messages,
    validate_model_name,
    validate_prompt_length,
    validate_response_format,
    validate_temperature,
)
from .config import OpenAIConfig


class OpenAIClient:
    """
    Main OpenAI client for handling API requests.

    This class provides a comprehensive interface for interacting with OpenAI's API,
    supporting both async and sync operations with robust error handling and retry logic.

    Features:
    - Async and sync API support
    - Automatic retry with exponential backoff
    - Comprehensive error handling
    - Request/response logging
    - Connection pooling
    - Streaming support
    - Context manager support
    """

    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        use_async: bool = True,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        enable_logging: bool = True,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            config: Configuration object. If None, will load from environment.
            use_async: Whether to use async client (default: True)
            timeout: Request timeout in seconds (overrides config)
            max_retries: Maximum retries for failed requests (overrides config)
            enable_logging: Whether to enable request/response logging
        """
        self.config = config or OpenAIConfig.from_env()
        self.use_async = use_async
        self.timeout = timeout or self.config.timeout
        self.max_retries = max_retries or self.config.max_retries
        self.enable_logging = enable_logging

        # Validate configuration
        self._validate_config()

        # Set up logging
        if self.enable_logging:
            logging.basicConfig(level=getattr(logging, self.config.log_level))
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

        # Initialize OpenAI client
        self._initialize_client()

        # Usage tracking
        self._usage_stats = {
            "requests": 0,
            "tokens_used": 0,
            "errors": 0,
            "start_time": time.time(),
        }

        self.logger.info(
            f"OpenAI client initialized with base URL: {self.config.base_url}"
        )

    def _validate_config(self) -> None:
        """Validate client configuration."""
        if not self.config.api_key:
            raise OpenAIValidationError("API key is required")

        if self.timeout and self.timeout <= 0:
            raise OpenAIValidationError("Timeout must be positive")

        if self.max_retries and self.max_retries < 0:
            raise OpenAIValidationError("Max retries must be non-negative")

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client with proper error handling."""
        try:
            # Check if OpenAI package is available
            _ = AsyncOpenAI
            _ = OpenAI
        except NameError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        if self.use_async:
            # Create async HTTP client with connection pooling
            http_client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )

            # Prepare client kwargs - only include organization if it's set and not empty
            client_kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "http_client": http_client,
            }

            # Only add organization if it's set and not empty
            if self.config.organization and self.config.organization.strip():
                client_kwargs["organization"] = self.config.organization

            self._client = AsyncOpenAI(**client_kwargs)
        else:
            # Create sync HTTP client with connection pooling
            http_client = httpx.Client(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )

            # Prepare client kwargs - only include organization if it's set and not empty
            client_kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "http_client": http_client,
            }

            # Only add organization if it's set and not empty
            if self.config.organization and self.config.organization.strip():
                client_kwargs["organization"] = self.config.organization

            self._client = OpenAI(**client_kwargs)

    def _should_not_retry(self, error: Exception) -> bool:
        """Determine if an error should not be retried."""
        if hasattr(error, "status_code"):
            status_code = getattr(error, "status_code")
            # Don't retry client errors (4xx) except rate limits
            if status_code in [400, 401, 403, 404, 422]:
                return True
            # Retry rate limits and server errors
            if status_code in [429, 500, 502, 503, 504]:
                return False

        # Don't retry validation errors
        if isinstance(error, (OpenAIValidationError, ValueError)):
            return True

        return False

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff time with jitter."""
        base_delay = 2**attempt  # Exponential backoff: 1, 2, 4, 8 seconds
        jitter = base_delay * 0.1 * (1 + (time.time() % 1))  # Add 10% jitter
        return min(base_delay + jitter, 60.0)  # Cap at 60 seconds

    def _handle_api_error(self, error: Exception) -> OpenAIError:
        """
        Handle API errors and convert them to appropriate exception types.

        Args:
            error: The original exception

        Returns:
            Appropriate OpenAI exception
        """
        if hasattr(error, "status_code"):
            status_code = getattr(error, "status_code")

            if status_code == 429:
                return OpenAIRateLimitError(f"Rate limit exceeded: {str(error)}")
            elif status_code == 402:
                return OpenAIQuotaError(f"Quota exceeded: {str(error)}")
            elif status_code >= 500:
                return OpenAIAPIError(f"Server error ({status_code}): {str(error)}")
            else:
                return OpenAIAPIError(f"API error ({status_code}): {str(error)}")

        return OpenAIAPIError(f"API request failed: {str(error)}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for completion
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Chat completion response

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if not self.use_async:
            raise RuntimeError(
                "Client is configured for sync use. Use sync_chat_completion() instead."
            )

        # Validate inputs
        validate_messages(messages)
        validate_model_name(model)
        validate_temperature(temperature)
        if max_tokens:
            validate_max_tokens(max_tokens)

        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        self.logger.info(f"Creating chat completion with model: {model}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = await self._client.chat.completions.create(**data)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Chat completion - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Update usage stats if available
                if hasattr(response, "usage") and response.usage:
                    self._usage_stats["tokens_used"] += getattr(
                        response.usage, "total_tokens", 0
                    )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Chat completion failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retrying (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Chat completion failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Chat completion failed with unknown error")

        raise self._handle_api_error(last_exception)

    async def create_embedding(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-ada-002",
        **kwargs: Any,
    ) -> Any:
        """
        Create embeddings for text.

        Args:
            input_text: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            Embedding response

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if not self.use_async:
            raise RuntimeError(
                "Client is configured for sync use. Use sync_create_embedding() instead."
            )

        # Validate inputs
        if isinstance(input_text, str):
            validate_prompt_length(input_text, max_length=8192)
        elif isinstance(input_text, list):
            for text in input_text:
                validate_prompt_length(text, max_length=8192)

        validate_model_name(model)

        data = {
            "input": input_text,
            "model": model,
            **kwargs,
        }

        self.logger.info(f"Creating embeddings with model: {model}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = await self._client.embeddings.create(**data)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Embedding creation - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Embedding creation failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retrying (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Embedding creation failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Embedding creation failed with unknown error")

        raise self._handle_api_error(last_exception)

    async def generate_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs: Any,
    ) -> Any:
        """
        Generate images from text prompt.

        Args:
            prompt: Text description of the image
            n: Number of images to generate (1-10)
            size: Image size
            response_format: Response format (url or b64_json)
            **kwargs: Additional parameters

        Returns:
            Image generation response

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if not self.use_async:
            raise RuntimeError(
                "Client is configured for sync use. Use sync_generate_image() instead."
            )

        # Validate inputs
        validate_prompt_length(prompt, max_length=1000)
        validate_image_size(size)
        validate_response_format(response_format)

        if not 1 <= n <= 10:
            raise OpenAIValidationError("Number of images must be between 1 and 10")

        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            **kwargs,
        }

        self.logger.info(f"Generating {n} image(s) with size: {size}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = await self._client.images.generate(**data)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Image generation - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Image generation failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retrying (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Image generation failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Image generation failed with unknown error")

        raise self._handle_api_error(last_exception)

    async def list_models(self) -> Any:
        """
        List available models.

        Returns:
            List of available models

        Raises:
            OpenAIAPIError: If API request fails
        """
        if not self.use_async:
            raise RuntimeError(
                "Client is configured for sync use. Use sync_list_models() instead."
            )

        self.logger.info("Listing available models")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = await self._client.models.list()

                response_time = time.time() - start_time
                self.logger.info(
                    f"List models - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"List models failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retrying (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"List models failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("List models failed with unknown error")

        raise self._handle_api_error(last_exception)

    async def get_model(self, model_id: str) -> Any:
        """
        Get information about a specific model.

        Args:
            model_id: ID of the model to retrieve

        Returns:
            Model information

        Raises:
            OpenAIValidationError: If model_id is invalid
            OpenAIAPIError: If API request fails
        """
        if not self.use_async:
            raise RuntimeError(
                "Client is configured for sync use. Use sync_get_model() instead."
            )

        validate_model_name(model_id)

        self.logger.info(f"Getting model information for: {model_id}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = await self._client.models.retrieve(model_id)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Get model - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Get model failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retrying (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Get model failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Get model failed with unknown error")

        raise self._handle_api_error(last_exception)

    # Sync methods
    def sync_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a chat completion (sync version).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for completion
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Chat completion response

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if self.use_async:
            raise RuntimeError(
                "Client is configured for async use. Use chat_completion() instead."
            )

        # Validate inputs
        validate_messages(messages)
        validate_model_name(model)
        validate_temperature(temperature)
        if max_tokens:
            validate_max_tokens(max_tokens)

        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        self.logger.info(f"Creating chat completion with model: {model}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = self._client.chat.completions.create(**data)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Chat completion - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Update usage stats if available
                if hasattr(response, "usage") and response.usage:
                    self._usage_stats["tokens_used"] += getattr(
                        response.usage, "total_tokens", 0
                    )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Chat completion failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retrying (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Chat completion failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Chat completion failed with unknown error")

        raise self._handle_api_error(last_exception)

    def sync_create_embedding(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-ada-002",
        **kwargs: Any,
    ) -> Any:
        """
        Create embeddings for text (sync version).

        Args:
            input_text: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            Embedding response

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if self.use_async:
            raise RuntimeError(
                "Client is configured for async use. Use create_embedding() instead."
            )

        # Validate inputs
        if isinstance(input_text, str):
            validate_prompt_length(input_text, max_length=8192)
        elif isinstance(input_text, list):
            for text in input_text:
                validate_prompt_length(text, max_length=8192)

        validate_model_name(model)

        data = {
            "input": input_text,
            "model": model,
            **kwargs,
        }

        self.logger.info(f"Creating embeddings with model: {model}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = self._client.embeddings.create(**data)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Embedding creation - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Embedding creation failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retry (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Embedding creation failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Embedding creation failed with unknown error")

        raise self._handle_api_error(last_exception)

    def sync_generate_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs: Any,
    ) -> Any:
        """
        Generate images from text prompt (sync version).

        Args:
            prompt: Text description of the image
            n: Number of images to generate (1-10)
            size: Image size
            response_format: Response format (url or b64_json)
            **kwargs: Additional parameters

        Returns:
            Image generation response

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if self.use_async:
            raise RuntimeError(
                "Client is configured for async use. Use generate_image() instead."
            )

        # Validate inputs
        validate_prompt_length(prompt, max_length=1000)
        validate_image_size(size)
        validate_response_format(response_format)

        if not 1 <= n <= 10:
            raise OpenAIValidationError("Number of images must be between 1 and 10")

        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            **kwargs,
        }

        self.logger.info(f"Generating {n} image(s) with size: {size}")

        # Execute with retry logic
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._usage_stats["requests"] += 1

                response = self._client.images.generate(**data)

                response_time = time.time() - start_time
                self.logger.info(
                    f"Image generation - Status: 200 - Time: {response_time:.3f}s - "
                    f"Attempt: {attempt + 1}"
                )

                # Handle different response types
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                elif hasattr(response, "dict"):
                    return response.dict()
                else:
                    return response

            except Exception as e:
                last_exception = e
                self._usage_stats["errors"] += 1

                self.logger.warning(
                    f"Image generation failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                )

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    break

                # Wait before retry (exponential backoff with jitter)
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        # If we get here, all retries failed
        total_time = time.time() - start_time
        self.logger.error(
            f"Image generation failed after {self.max_retries + 1} attempts in {total_time:.3f}s"
        )

        if last_exception is None:
            raise OpenAIError("Image generation failed with unknown error")

        raise self._handle_api_error(last_exception)

    # Streaming methods
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completion responses.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for completion
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Streaming response chunks

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if not self.use_async:
            raise RuntimeError(
                "Client is configured for sync use. Use sync_stream_chat_completion() instead."
            )

        # Validate inputs
        validate_messages(messages)
        validate_model_name(model)
        validate_temperature(temperature)
        if max_tokens:
            validate_max_tokens(max_tokens)

        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        self.logger.info(f"Starting streaming chat completion with model: {model}")

        try:
            stream = await self._client.chat.completions.create(**data)
            async for chunk in stream:
                if hasattr(chunk, "model_dump"):
                    yield chunk.model_dump()
                else:
                    yield chunk
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            raise self._handle_api_error(e)

    def sync_stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion responses (sync version).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for completion
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Streaming response chunks

        Raises:
            OpenAIValidationError: If input validation fails
            OpenAIAPIError: If API request fails
        """
        if self.use_async:
            raise RuntimeError(
                "Client is configured for async use. Use stream_chat_completion() instead."
            )

        # Validate inputs
        validate_messages(messages)
        validate_model_name(model)
        validate_temperature(temperature)
        if max_tokens:
            validate_max_tokens(max_tokens)

        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        self.logger.info(f"Starting streaming chat completion with model: {model}")

        try:
            stream = self._client.chat.completions.create(**data)
            for chunk in stream:
                if hasattr(chunk, "model_dump"):
                    yield chunk.model_dump()
                else:
                    yield chunk
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            raise self._handle_api_error(e)

    # Utility methods
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        uptime = time.time() - self._usage_stats["start_time"]
        return {
            **self._usage_stats,
            "uptime_seconds": uptime,
            "requests_per_minute": (
                (self._usage_stats["requests"] / uptime) * 60 if uptime > 0 else 0
            ),
            "error_rate": (
                self._usage_stats["errors"] / self._usage_stats["requests"]
                if self._usage_stats["requests"] > 0
                else 0
            ),
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._usage_stats = {
            "requests": 0,
            "tokens_used": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    def close(self) -> None:
        """Close the client and clean up resources."""
        # Close the underlying HTTP client if it exists
        if hasattr(self._client, "_client"):
            if hasattr(self._client._client, "close"):
                self._client._client.close()
        self.logger.info("OpenAI client closed")

    async def aclose(self) -> None:
        """Close the client asynchronously and clean up resources."""
        # Close the underlying HTTP client if it exists
        if hasattr(self._client, "_client"):
            if hasattr(self._client._client, "aclose"):
                await self._client._client.aclose()
        self.logger.info("OpenAI client closed")

    # Context manager support
    def __enter__(self) -> "OpenAIClient":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    async def __aenter__(self) -> "OpenAIClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.aclose()

    # Session management
    @contextmanager
    def session(self) -> Iterator["OpenAIClient"]:
        """Create a session context manager."""
        try:
            yield self
        finally:
            self.close()

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator["OpenAIClient", None]:
        """Create an async session context manager."""
        try:
            yield self
        finally:
            await self.aclose()
