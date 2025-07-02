"""Logging utilities for the OpenAI client."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "openai_client",
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_to_file: bool = False,
    log_file: str = "openai_client.log",
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        log_to_file: Whether to log to file
        log_file: Log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set log level
    logger.setLevel(getattr(logging, level.upper()))

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "openai_client") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """
    Log function call with parameters.

    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters
    """
    # Filter out sensitive information
    safe_kwargs = {}
    for key, value in kwargs.items():
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            safe_kwargs[key] = "***"
        else:
            safe_kwargs[key] = value

    logger.debug(f"Calling {func_name} with parameters: {safe_kwargs}")


def log_api_response(
    logger: logging.Logger, endpoint: str, status_code: int, response_time: float
) -> None:
    """
    Log API response information.

    Args:
        logger: Logger instance
        endpoint: API endpoint
        status_code: HTTP status code
        response_time: Response time in seconds
    """
    logger.info(f"API {endpoint} - Status: {status_code} - Time: {response_time:.3f}s")
