# OpenAI Client

A comprehensive, well-structured OpenAI client with advanced features, following SOLID principles and modern Python development practices.

## Features

- **Modular Design**: Clean separation of concerns with interfaces and implementations
- **Async Support**: Full async/await support for high-performance operations
- **Type Safety**: Comprehensive type annotations with Pydantic models
- **CLI Interface**: Command-line interface for easy interaction
- **Configuration Management**: Environment-based configuration with validation
- **Error Handling**: Robust error handling with custom exceptions
- **Testing**: Comprehensive test suite with pytest
- **Code Quality**: Pre-commit hooks, linting, and formatting

## Project Structure

```
openai_client/
├── src/
│   └── openai_client/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── client.py          # Main client interface
│       │   ├── config.py          # Configuration management
│       │   └── exceptions.py      # Custom exceptions
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py        # Request models
│       │   └── responses.py       # Response models
│       ├── services/
│       │   ├── __init__.py
│       │   ├── chat_service.py    # Chat completion service
│       │   ├── embedding_service.py # Embedding service
│       │   └── image_service.py   # Image generation service
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logger.py          # Logging utilities
│       │   └── validators.py      # Input validation
│       └── cli.py                 # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_client.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_chat_service.py
│   │   ├── test_embedding_service.py
│   │   └── test_image_service.py
│   └── test_models/
│       ├── __init__.py
│       ├── test_requests.py
│       └── test_responses.py
├── .env.example
├── .gitignore
├── environment.yml
├── pyproject.toml
├── pre-commit-config.yaml
└── README.md
```

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate openai-client
```

### 2. Install in Development Mode

```bash
pip install -e .
```

### 3. Set up Pre-commit Hooks

```bash
pre-commit install
```

### 4. Configure Environment

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### As a Library

```python
from openai_client import OpenAIClient
from openai_client.models.requests import ChatRequest

# Initialize client
client = OpenAIClient()

# Create chat request
request = ChatRequest(
    messages=[{"role": "user", "content": "Hello, world!"}],
    model="gpt-3.5-turbo"
)

# Get response
response = await client.chat.complete(request)
print(response.choices[0].message.content)
```

### Command Line Interface

```bash
# Chat completion
openai-client chat "Hello, how are you?"

# Generate embeddings
openai-client embed "This is a test sentence"

# Generate image
openai-client image "A beautiful sunset over mountains"
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
flake8 src/
```

## Design Patterns Used

- **Strategy Pattern**: Different service implementations for different OpenAI endpoints
- **Factory Pattern**: Client factory for creating different types of clients
- **Singleton Pattern**: Configuration management
- **Observer Pattern**: Event handling for streaming responses
- **Adapter Pattern**: Adapting different API response formats

## SOLID Principles

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to extend with new services without modifying existing code
- **Liskov Substitution**: All service implementations are interchangeable
- **Interface Segregation**: Clients depend only on the interfaces they use
- **Dependency Inversion**: High-level modules don't depend on low-level modules

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
