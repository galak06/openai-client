import os

import pytest
from dotenv import load_dotenv

from openai_client.core.client import OpenAIClient


@pytest.mark.asyncio
async def test_client_with_env_api_key():
    # Load environment variables from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set in your .env file."

    # Check if organization is set and provide guidance
    organization = os.getenv("OPENAI_ORGANIZATION")
    if not organization:
        print(
            "Warning: OPENAI_ORGANIZATION not set. If your API key is associated with an organization, you may need to set this."
        )

    client = OpenAIClient()
    try:
        # Test listing models
        models = await client.list_models()
        assert "data" in models
        assert isinstance(models["data"], list)
        print(f"Found {len(models['data'])} models.")

        # Test sending a chat message
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with a simple greeting."}
        ]

        response = await client.chat_completion(
            messages=test_messages,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50,
        )

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]

        print(f"Chat response: {response['choices'][0]['message']['content']}")

    except Exception as e:
        if "mismatched_organization" in str(e):
            print(
                "Error: Organization mismatch. Please check your OPENAI_ORGANIZATION setting."
            )
            print("If using a personal API key, leave OPENAI_ORGANIZATION unset.")
            print(
                "If using an organization API key, set OPENAI_ORGANIZATION to the correct organization ID."
            )
        raise
    finally:
        await client.aclose()
