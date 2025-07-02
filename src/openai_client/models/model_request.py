"""Model request for OpenAI API."""

from pydantic import BaseModel


class ModelRequest(BaseModel):
    """
    Request model for listing available models.

    This model is used for the models list API endpoint.
    """

    # This is a simple request model as the models endpoint
    # typically doesn't require parameters
    pass
