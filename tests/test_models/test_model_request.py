"""Unit tests for ModelRequest model."""

import pytest
from pydantic import ValidationError

from openai_client.models.model_request import ModelRequest


class TestModelRequest:
    """Test cases for ModelRequest model."""

    def test_valid_model_request(self):
        """Test creating a valid model request."""
        request = ModelRequest()

        # ModelRequest has no fields, so it should be valid
        assert request is not None

    def test_model_dump(self):
        """Test model serialization."""
        request = ModelRequest()
        data = request.model_dump()

        # Should return empty dict since no fields
        assert data == {}

    def test_model_dump_json(self):
        """Test model JSON serialization."""
        request = ModelRequest()
        json_data = request.model_dump_json()

        # Should return empty JSON object
        assert json_data == "{}"

    def test_equality(self):
        """Test request equality."""
        request1 = ModelRequest()
        request2 = ModelRequest()

        assert request1 == request2

    def test_repr(self):
        """Test request string representation."""
        request = ModelRequest()
        repr_str = repr(request)

        assert "ModelRequest" in repr_str

    def test_str(self):
        """Test request string conversion."""
        request = ModelRequest()
        str_repr = str(request)

        # ModelRequest is an empty class, so str() returns empty string
        assert str_repr == ""
