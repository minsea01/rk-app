#!/usr/bin/env python3
"""Unified logging configuration for rk-app.

This module provides consistent logging setup across all modules,
making it easier to control log levels and output formats.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name (usually __name__ of the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        console: Whether to output logs to console

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting inference...")
        >>> logger.warning("Low confidence detection")
        >>> logger.error("Failed to load model")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Convenience functions for common logging levels
def set_log_level(logger: logging.Logger, level: int):
    """Set the log level for a logger and all its handlers.

    Args:
        logger: Logger instance
        level: New logging level
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def enable_debug(logger: logging.Logger):
    """Enable debug logging.

    Args:
        logger: Logger instance
    """
    set_log_level(logger, logging.DEBUG)


def disable_debug(logger: logging.Logger):
    """Disable debug logging (set to INFO).

    Args:
        logger: Logger instance
    """
    set_log_level(logger, logging.INFO)
