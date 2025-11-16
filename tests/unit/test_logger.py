#!/usr/bin/env python3
"""Unit tests for logger module."""
import pytest
import logging
from pathlib import Path
import tempfile
import sys

from apps.logger import (
    setup_logger,
    get_logger,
    set_log_level,
    enable_debug,
    disable_debug
)


class TestSetupLogger:
    """Test suite for setup_logger function."""

    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a valid logger."""
        logger = setup_logger('test_basic')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_basic'

    def test_setup_logger_default_level(self):
        """Test logger is created with default INFO level."""
        logger = setup_logger('test_level_default')
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level(self):
        """Test logger can be created with custom level."""
        logger = setup_logger('test_level_debug', level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logger_has_console_handler(self):
        """Test logger has console handler by default."""
        logger = setup_logger('test_console')
        # Should have at least one StreamHandler
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) > 0

    def test_setup_logger_no_console_handler(self):
        """Test logger can be created without console handler."""
        logger = setup_logger('test_no_console', console=False)
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        ]
        assert len(stream_handlers) == 0

    def test_setup_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            logger = setup_logger('test_file', log_file=log_file)
            logger.info('test message')

            # Check file was created
            assert log_file.exists()
            # Check message was written
            content = log_file.read_text()
            assert 'test message' in content

    def test_setup_logger_file_directory_created(self):
        """Test logger creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'subdir' / 'nested' / 'test.log'
            logger = setup_logger('test_mkdir', log_file=log_file)
            logger.info('test')

            assert log_file.exists()
            assert log_file.parent.exists()

    def test_setup_logger_format_includes_timestamp(self):
        """Test log format includes timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'format_test.log'
            logger = setup_logger('test_format', log_file=log_file)
            logger.info('format test message')

            content = log_file.read_text()
            # Should contain timestamp pattern (YYYY-MM-DD HH:MM:SS)
            assert '-' in content  # Date separator
            assert ':' in content  # Time separator
            assert 'test_format' in content  # Logger name
            assert 'INFO' in content  # Log level
            assert 'format test message' in content


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_creates_new(self):
        """Test get_logger creates new logger if not exists."""
        logger = get_logger('test_new_logger')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_new_logger'

    def test_get_logger_reuses_existing(self):
        """Test get_logger returns existing logger."""
        logger1 = setup_logger('test_reuse')
        logger2 = get_logger('test_reuse')
        # Should be the same logger instance
        assert logger1 is logger2

    def test_get_logger_has_handlers(self):
        """Test get_logger ensures logger has handlers."""
        logger = get_logger('test_has_handlers')
        assert len(logger.handlers) > 0


class TestSetLogLevel:
    """Test suite for set_log_level function."""

    def test_set_log_level_changes_level(self):
        """Test set_log_level changes logger level."""
        logger = setup_logger('test_change_level', level=logging.INFO)
        set_log_level(logger, logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_set_log_level_changes_handler_level(self):
        """Test set_log_level changes all handler levels."""
        logger = setup_logger('test_handler_level', level=logging.INFO)
        set_log_level(logger, logging.WARNING)

        for handler in logger.handlers:
            assert handler.level == logging.WARNING


class TestEnableDebug:
    """Test suite for enable_debug function."""

    def test_enable_debug_sets_debug_level(self):
        """Test enable_debug sets logger to DEBUG level."""
        logger = setup_logger('test_enable_debug', level=logging.INFO)
        enable_debug(logger)
        assert logger.level == logging.DEBUG

    def test_enable_debug_affects_handlers(self):
        """Test enable_debug sets handler levels to DEBUG."""
        logger = setup_logger('test_enable_debug_handlers', level=logging.INFO)
        enable_debug(logger)

        for handler in logger.handlers:
            assert handler.level == logging.DEBUG


class TestDisableDebug:
    """Test suite for disable_debug function."""

    def test_disable_debug_sets_info_level(self):
        """Test disable_debug sets logger to INFO level."""
        logger = setup_logger('test_disable_debug', level=logging.DEBUG)
        disable_debug(logger)
        assert logger.level == logging.INFO

    def test_disable_debug_affects_handlers(self):
        """Test disable_debug sets handler levels to INFO."""
        logger = setup_logger('test_disable_debug_handlers', level=logging.DEBUG)
        disable_debug(logger)

        for handler in logger.handlers:
            assert handler.level == logging.INFO


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_logger_writes_to_file_and_console(self):
        """Test logger can write to both file and console."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'both.log'
            logger = setup_logger('test_both', log_file=log_file, console=True)

            test_msg = 'dual output test'
            logger.info(test_msg)

            # Check file output
            assert log_file.exists()
            assert test_msg in log_file.read_text()

            # Check has console handler
            stream_handlers = [
                h for h in logger.handlers
                if isinstance(h, logging.StreamHandler)
            ]
            assert len(stream_handlers) > 0

    def test_logger_level_filtering(self):
        """Test logger filters messages by level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'filter.log'
            logger = setup_logger('test_filter', level=logging.WARNING, log_file=log_file)

            logger.debug('debug message')
            logger.info('info message')
            logger.warning('warning message')
            logger.error('error message')

            content = log_file.read_text()
            # Should not contain debug/info
            assert 'debug message' not in content
            assert 'info message' not in content
            # Should contain warning/error
            assert 'warning message' in content
            assert 'error message' in content
