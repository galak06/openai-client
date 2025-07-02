#!/usr/bin/env python3
"""
Test script for the updated OpenAIClient.
"""

import asyncio
import os

import pytest

from openai_client.core.client import OpenAIClient
from openai_client.core.config import OpenAIConfig


@pytest.mark.asyncio
async def test_async_client():
    """Test async client functionality."""
    print("Testing Async Client...")

    # Create config with test API key
    config = OpenAIConfig(
        api_key="sk-test123456789012345678901234567890123456789012345678901234567890",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=3,
        log_level="INFO",
    )

    # Initialize async client
    client = OpenAIClient(config=config, use_async=True, enable_logging=True)

    try:
        # Test usage stats
        stats = client.get_usage_stats()
        print(f"Initial usage stats: {stats}")

        # Test context manager
        async with client.async_session() as session:
            print("Async session started")

            # Test list models (this will fail with test key, but we can test the error handling)
            try:
                models = await session.list_models()
                print(f"Models: {models}")
            except Exception as e:
                print(f"Expected error for list_models: {type(e).__name__}: {e}")

        print("Async session completed")

    except Exception as e:
        print(f"Error in async client test: {e}")
    finally:
        await client.aclose()


def test_sync_client():
    """Test sync client functionality."""
    print("\nTesting Sync Client...")

    # Create config with test API key
    config = OpenAIConfig(
        api_key="sk-test123456789012345678901234567890123456789012345678901234567890",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=3,
        log_level="INFO",
    )

    # Initialize sync client
    client = OpenAIClient(config=config, use_async=False, enable_logging=True)

    try:
        # Test usage stats
        stats = client.get_usage_stats()
        print(f"Initial usage stats: {stats}")

        # Test context manager
        with client.session() as session:
            print("Sync session started")

            # Test chat completion (this will fail with test key, but we can test the error handling)
            try:
                response = session.sync_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-3.5-turbo",
                )
                print(f"Response: {response}")
            except Exception as e:
                print(
                    f"Expected error for sync_chat_completion: {type(e).__name__}: {e}"
                )

        print("Sync session completed")

    except Exception as e:
        print(f"Error in sync client test: {e}")
    finally:
        client.close()


@pytest.mark.asyncio
async def test_client_features():
    """Test client features and configuration."""
    print("\nTesting Client Features...")

    # Test configuration validation
    try:
        # This should raise an error for invalid timeout
        config = OpenAIConfig(
            api_key="sk-test123456789012345678901234567890123456789012345678901234567890",
            timeout=-1,  # Invalid timeout
            max_retries=3,
        )
        client = OpenAIClient(config=config)
    except Exception as e:
        print(f"Expected validation error: {type(e).__name__}: {e}")

    # Test valid configuration
    config = OpenAIConfig(
        api_key="sk-test123456789012345678901234567890123456789012345678901234567890",
        timeout=30,
        max_retries=3,
        log_level="WARNING",
    )

    client = OpenAIClient(config=config, use_async=True, enable_logging=False)

    # Test usage stats
    stats = client.get_usage_stats()
    print(f"Usage stats: {stats}")

    # Test reset stats
    client.reset_usage_stats()
    stats = client.get_usage_stats()
    print(f"Reset usage stats: {stats}")

    await client.aclose()


async def main():
    """Main test function."""
    print("Testing Updated OpenAIClient")
    print("=" * 50)

    # Test async client
    await test_async_client()

    # Test sync client
    test_sync_client()

    # Test client features
    await test_client_features()

    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
