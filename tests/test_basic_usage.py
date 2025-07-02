#!/usr/bin/env python3
"""
Basic usage example for the OpenAI client.

This example demonstrates how to use the OpenAI client for chat completions,
embeddings, and image generation.
"""

import asyncio
import os
from typing import List

from openai_client import OpenAIClient, OpenAIConfig


async def chat_completion_example(client: OpenAIClient) -> None:
    """Example of using chat completion."""
    print("=== Chat Completion Example ===")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What can you help me with today?"},
    ]

    try:
        response = await client.chat_completion(
            messages=messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150
        )

        print(f"Response: {response['choices'][0]['message']['content']}")
        print(f"Usage: {response['usage']}")

    except Exception as e:
        print(f"Error in chat completion: {e}")


async def embedding_example(client: OpenAIClient) -> None:
    """Example of creating embeddings."""
    print("\n=== Embedding Example ===")

    text = "This is a sample text for embedding generation."

    try:
        response = await client.create_embedding(
            input_text=text, model="text-embedding-ada-002"
        )

        embedding = response["data"][0]["embedding"]
        print(f"Embedding length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

    except Exception as e:
        print(f"Error in embedding: {e}")


async def image_generation_example(client: OpenAIClient) -> None:
    """Example of image generation."""
    print("\n=== Image Generation Example ===")

    prompt = "A beautiful sunset over mountains, digital art style"

    try:
        response = await client.generate_image(
            prompt=prompt, n=1, size="512x512", response_format="url"
        )

        image_url = response["data"][0]["url"]
        print(f"Generated image URL: {image_url}")

    except Exception as e:
        print(f"Error in image generation: {e}")


async def list_models_example(client: OpenAIClient) -> None:
    """Example of listing available models."""
    print("\n=== List Models Example ===")

    try:
        response = await client.list_models()

        print("Available models:")
        for model in response["data"][:5]:  # Show first 5 models
            print(f"- {model['id']} ({model['object']})")

    except Exception as e:
        print(f"Error listing models: {e}")


async def main() -> None:
    """Main function demonstrating all examples."""
    print("OpenAI Client Examples")
    print("=" * 50)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in the .env file or environment.")
        return

    # Initialize client
    try:
        client = OpenAIClient()
        print("Client initialized successfully!")

        # Run examples
        await chat_completion_example(client)
        await embedding_example(client)
        await image_generation_example(client)
        await list_models_example(client)

        # Clean up
        await client.aclose()

    except Exception as e:
        print(f"Error initializing client: {e}")


if __name__ == "__main__":
    asyncio.run(main())
