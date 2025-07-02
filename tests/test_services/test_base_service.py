"""Unit tests for BaseService class."""

from unittest.mock import AsyncMock, Mock

import pytest

from openai_client.exceptions import OpenAIError
from openai_client.services.base_service import BaseService


class TestBaseService:
    """Test cases for BaseService class."""

    def test_init(self):
        """Test BaseService initialization."""
        mock_client = Mock()
        mock_client.logger = Mock()

        # Create a concrete implementation for testing
        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        assert service.client == mock_client
        assert service.logger == mock_client.logger

    def test_validate_required_params_success(self):
        """Test validate_required_params with all required params present."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        params = {"param1": "value1", "param2": "value2"}
        required = ["param1", "param2"]

        # Should not raise an exception
        service.validate_required_params(params, required)

    def test_validate_required_params_missing(self):
        """Test validate_required_params with missing required params."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        params = {"param1": "value1"}
        required = ["param1", "param2"]

        with pytest.raises(OpenAIError) as exc_info:
            service.validate_required_params(params, required)

        assert "Missing required parameters: param2" in str(exc_info.value)

    def test_validate_required_params_none_value(self):
        """Test validate_required_params with None value."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        params = {"param1": "value1", "param2": None}
        required = ["param1", "param2"]

        with pytest.raises(OpenAIError) as exc_info:
            service.validate_required_params(params, required)

        assert "Missing required parameters: param2" in str(exc_info.value)

    def test_validate_required_params_empty_list(self):
        """Test validate_required_params with empty required list."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        params = {"param1": "value1"}
        required = []

        # Should not raise an exception
        service.validate_required_params(params, required)

    def test_log_operation(self):
        """Test log_operation method."""
        mock_client = Mock()
        mock_logger = Mock()
        mock_client.logger = mock_logger

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        service.log_operation("test_operation", param1="value1", param2="value2")

        mock_logger.info.assert_called_once_with(
            "Executing test_operation: {'param1': 'value1', 'param2': 'value2'}"
        )

    def test_log_operation_no_kwargs(self):
        """Test log_operation method with no additional kwargs."""
        mock_client = Mock()
        mock_logger = Mock()
        mock_client.logger = mock_logger

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        service.log_operation("test_operation")

        mock_logger.info.assert_called_once_with("Executing test_operation: {}")

    def test_abstract_method(self):
        """Test that BaseService cannot be instantiated directly."""
        mock_client = Mock()
        mock_client.logger = Mock()

        with pytest.raises(TypeError):
            BaseService(mock_client)  # type: ignore[abstract]

    def test_execute_abstract_method(self):
        """Test that execute method is abstract."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            pass  # Don't implement execute

        with pytest.raises(TypeError):
            TestService(mock_client)  # type: ignore[abstract]

    def test_validate_required_params_multiple_missing(self):
        """Test validate_required_params with multiple missing params."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        params = {"param1": "value1"}
        required = ["param1", "param2", "param3"]

        with pytest.raises(OpenAIError) as exc_info:
            service.validate_required_params(params, required)

        assert "Missing required parameters: param2, param3" in str(exc_info.value)

    def test_validate_required_params_empty_params(self):
        """Test validate_required_params with empty params dict."""
        mock_client = Mock()
        mock_client.logger = Mock()

        class TestService(BaseService):
            async def execute(self, **kwargs):
                return {"test": "result"}

        service = TestService(mock_client)

        params = {}
        required = ["param1", "param2"]

        with pytest.raises(OpenAIError) as exc_info:
            service.validate_required_params(params, required)

        assert "Missing required parameters: param1, param2" in str(exc_info.value)
