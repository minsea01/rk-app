#!/usr/bin/env python3
"""Unit tests for tools.http_receiver."""

import json
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from apps.exceptions import ConfigurationError
from tools.http_receiver import Handler, main


def _make_handler(body: bytes, content_length: str) -> Handler:
    handler = Handler(MagicMock(), ("127.0.0.1", 12345), None)
    handler.headers = {"content-length": content_length}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    return handler


class TestHTTPReceiverHandler:
    def test_do_post_accepts_valid_json(self):
        handler = _make_handler(b'{"test":"data","id":1}', "22")

        handler.do_POST()

        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call("Content-Type", "application/json")
        assert json.loads(handler.wfile.getvalue().decode("utf-8")) == {"status": "ok"}

    def test_do_post_rejects_invalid_content_length(self):
        handler = _make_handler(b"{}", "invalid")

        handler.do_POST()

        handler.send_response.assert_called_once_with(400)
        assert b"Invalid Content-Length" in handler.wfile.getvalue()

    def test_do_post_rejects_payload_too_large(self):
        handler = _make_handler(b"{}", str(Handler.MAX_CONTENT_LENGTH + 1))

        handler.do_POST()

        handler.send_response.assert_called_once_with(413)
        assert b"Payload too large" in handler.wfile.getvalue()

    def test_do_post_handles_binary_payload_gracefully(self):
        handler = _make_handler(b"\x80\x81\x82\x83", "4")

        handler.do_POST()

        handler.send_response.assert_called_once_with(200)
        assert json.loads(handler.wfile.getvalue().decode("utf-8")) == {"status": "ok"}


class TestHTTPReceiverMain:
    def test_main_prints_selected_port_and_handles_keyboard_interrupt(self, capsys):
        mock_server = MagicMock()
        mock_server.server_address = ("127.0.0.1", 18081)
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        with patch("tools.http_receiver.HTTPServer", return_value=mock_server):
            with patch.object(sys, "argv", ["http_receiver.py", "--port", "0"]):
                assert main() == 0

        stdout = capsys.readouterr().out.strip()
        assert json.loads(stdout) == {"listening_port": 18081}
        mock_server.shutdown.assert_called_once()

    @pytest.mark.parametrize(
        ("errno_value", "message"),
        [
            (98, "already in use"),
            (13, "Permission denied"),
        ],
    )
    def test_main_wraps_bind_errors_as_configuration_error(self, errno_value, message):
        bind_error = OSError(errno_value, "bind failed")

        with patch("tools.http_receiver.HTTPServer", side_effect=bind_error):
            with patch.object(sys, "argv", ["http_receiver.py", "--port", "8081"]):
                with pytest.raises(ConfigurationError, match=message):
                    main()
