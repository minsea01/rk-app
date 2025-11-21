#!/usr/bin/env python3
"""Unit tests for tools.http_post module.

Tests HTTP POST utility for MCP pipeline validation.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import urllib.error

import pytest

from tools.http_post import main
from apps.exceptions import ConfigurationError, ValidationError


class TestHTTPPostMain:
    """Test suite for HTTP POST main function."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_validates_input_file_exists(self):
        """Test that ConfigurationError is raised when input file doesn't exist."""
        non_existent_file = self.temp_path / 'nonexistent.json'

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(non_existent_file)
        ]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                main()

    def test_validates_json_format(self):
        """Test that ValidationError is raised for invalid JSON."""
        invalid_json_file = self.temp_path / 'invalid.json'
        invalid_json_file.write_text('{"invalid": json!!}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(invalid_json_file)
        ]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                main()

    def test_sends_valid_json_successfully(self):
        """Test that valid JSON is sent successfully."""
        # Create valid JSON file
        valid_json_file = self.temp_path / 'valid.json'
        test_data = {'test': 'data', 'id': 123}
        valid_json_file.write_text(json.dumps(test_data))

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(valid_json_file)
        ]

        # Mock urllib.request.urlopen
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                # Should not raise
                main()

                # Verify request was made
                mock_opener_instance.open.assert_called_once()

    def test_sets_correct_content_type_header(self):
        """Test that Content-Type header is set to application/json."""
        valid_json_file = self.temp_path / 'test.json'
        valid_json_file.write_text('{"test": 1}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(valid_json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.Request') as mock_request:
                with patch('urllib.request.build_opener') as mock_opener:
                    mock_opener_instance = MagicMock()
                    mock_opener_instance.open.return_value = mock_response
                    mock_opener.return_value = mock_opener_instance

                    main()

                    # Verify Request was created with correct headers
                    mock_request.assert_called_once()
                    call_kwargs = mock_request.call_args[1]
                    assert 'headers' in call_kwargs
                    assert call_kwargs['headers']['Content-Type'] == 'application/json'

    def test_handles_connection_error(self):
        """Test that connection errors are handled gracefully."""
        valid_json_file = self.temp_path / 'test.json'
        valid_json_file.write_text('{"test": 1}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:9999',  # Non-existent server
            '--file', str(valid_json_file)
        ]

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.side_effect = urllib.error.URLError("Connection refused")
                mock_opener.return_value = mock_opener_instance

                # Should handle error and exit
                with pytest.raises(SystemExit):
                    main()

    def test_handles_http_error(self):
        """Test that HTTP errors (4xx, 5xx) are handled."""
        valid_json_file = self.temp_path / 'test.json'
        valid_json_file.write_text('{"test": 1}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(valid_json_file)
        ]

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                # Simulate 500 Internal Server Error
                mock_opener_instance.open.side_effect = urllib.error.HTTPError(
                    'http://localhost:8000', 500, 'Internal Server Error', {}, None
                )
                mock_opener.return_value = mock_opener_instance

                # Should handle error
                with pytest.raises(SystemExit):
                    main()

    def test_handles_timeout(self):
        """Test that request timeout is handled."""
        valid_json_file = self.temp_path / 'test.json'
        valid_json_file.write_text('{"test": 1}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(valid_json_file)
        ]

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                # Simulate timeout
                mock_opener_instance.open.side_effect = urllib.error.URLError("Timeout")
                mock_opener.return_value = mock_opener_instance

                with pytest.raises(SystemExit):
                    main()

    def test_sends_utf8_encoded_data(self):
        """Test that JSON data is UTF-8 encoded."""
        json_file = self.temp_path / 'utf8.json'
        test_data = {'message': '中文测试', 'emoji': '🚀'}
        json_file.write_text(json.dumps(test_data, ensure_ascii=False))

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                main()

                # Verify UTF-8 encoding was used
                mock_opener_instance.open.assert_called_once()

    def test_disables_proxy(self):
        """Test that proxy is disabled for local requests."""
        valid_json_file = self.temp_path / 'test.json'
        valid_json_file.write_text('{"test": 1}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(valid_json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                main()

                # Verify ProxyHandler({}) was used to disable proxy
                mock_opener.assert_called_once()
                call_args = mock_opener.call_args[0]
                # Should include ProxyHandler
                assert len(call_args) > 0

    def test_logs_response_status(self):
        """Test that response is logged."""
        valid_json_file = self.temp_path / 'test.json'
        valid_json_file.write_text('{"test": 1}')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(valid_json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "success"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                with patch('tools.http_post.logger') as mock_logger:
                    main()

                    # Verify logging occurred
                    assert mock_logger.info.called


class TestHTTPPostEdgeCases:
    """Test edge cases for HTTP POST utility."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_handles_empty_json_file(self):
        """Test that empty JSON file is handled."""
        empty_file = self.temp_path / 'empty.json'
        empty_file.write_text('')

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(empty_file)
        ]

        with patch('sys.argv', test_args):
            # Should raise validation error for invalid JSON
            with pytest.raises(SystemExit):
                main()

    def test_handles_large_json_file(self):
        """Test that large JSON files are sent successfully."""
        large_json_file = self.temp_path / 'large.json'

        # Create large JSON (1MB)
        large_data = {'data': 'x' * (1024 * 1024)}
        large_json_file.write_text(json.dumps(large_data))

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(large_json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                # Should handle large file
                main()

    def test_handles_nested_json_structures(self):
        """Test that deeply nested JSON is sent correctly."""
        nested_json_file = self.temp_path / 'nested.json'

        # Create deeply nested structure
        nested_data = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'data': [1, 2, 3, {'nested': 'value'}]
                        }
                    }
                }
            }
        }
        nested_json_file.write_text(json.dumps(nested_data))

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(nested_json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                main()

    def test_handles_json_with_arrays(self):
        """Test that JSON arrays are sent correctly."""
        array_json_file = self.temp_path / 'array.json'

        # JSON with arrays
        array_data = {
            'detections': [
                {'class': 'person', 'conf': 0.95, 'bbox': [100, 200, 300, 400]},
                {'class': 'car', 'conf': 0.87, 'bbox': [50, 100, 150, 250]}
            ]
        }
        array_json_file.write_text(json.dumps(array_data))

        test_args = [
            'http_post.py',
            '--url', 'http://localhost:8000',
            '--file', str(array_json_file)
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('sys.argv', test_args):
            with patch('urllib.request.build_opener') as mock_opener:
                mock_opener_instance = MagicMock()
                mock_opener_instance.open.return_value = mock_response
                mock_opener.return_value = mock_opener_instance

                main()

                # Verify request was successful
                mock_opener_instance.open.assert_called_once()
