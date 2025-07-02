"""Base service class for OpenAI API operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.client import OpenAIClient
from ..exceptions import OpenAIError


class BaseService(ABC):
    """
    Base service class for OpenAI API operations.

    This abstract base class defines the interface that all service
    implementations must follow, following the Strategy Pattern.
    """

    def __init__(self, client: OpenAIClient) -> None:
        """
        Initialize the base service.

        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.logger = client.logger

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the service operation.

        Args:
            **kwargs: Service-specific parameters

        Returns:
            Service response

        Raises:
            OpenAIError: If the operation fails
        """
        pass

    def validate_required_params(self, params: Dict[str, Any], required: list) -> None:
        """
        Validate that required parameters are present.

        Args:
            params: Parameters to validate
            required: List of required parameter names

        Raises:
            OpenAIError: If required parameters are missing
        """
        missing = [
            param for param in required if param not in params or params[param] is None
        ]
        if missing:
            raise OpenAIError(f"Missing required parameters: {', '.join(missing)}")

    def log_operation(self, operation: str, **kwargs: Any) -> None:
        """
        Log operation details.

        Args:
            operation: Operation name
            **kwargs: Additional logging parameters
        """
        self.logger.info(f"Executing {operation}: {kwargs}")
