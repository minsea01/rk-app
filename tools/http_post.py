#!/usr/bin/env python3
"""HTTP POST utility for MCP pipeline validation.

This script sends JSON payloads to HTTP endpoints for testing
data ingestion and network communication.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Import custom exceptions and logger
from apps.exceptions import ConfigurationError, ValidationError
from apps.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level='INFO')


def _send_json(url: str, payload: bytes) -> str:
    """Send JSON payload to target URL and return response text."""
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    with opener.open(req, timeout=5) as resp:
        response_text = resp.read().decode('utf-8')
        logger.info(f"Response: {response_text}")
        print(response_text)
        return response_text


def _load_payload(file_path: Path) -> bytes:
    if not file_path.exists():
        raise ConfigurationError(f"Input file not found: {file_path}")

    logger.info(f"Reading payload from: {file_path}")
    try:
        raw_text = file_path.read_text(encoding='utf-8')
    except (IOError, OSError, UnicodeDecodeError) as e:
        raise ConfigurationError(f"Failed to read payload file: {e}") from e

    try:
        json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {file_path}: {e}") from e

    return raw_text.encode('utf-8')


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='POST JSON file to HTTP endpoint')
    ap.add_argument('--url', required=True, help='Target URL')
    ap.add_argument('--file', required=True, help='JSON file to send')
    args = ap.parse_args(argv)

    try:
        payload = _load_payload(Path(args.file))
        logger.info(f"Sending POST request to: {args.url}")
        _send_json(args.url, payload)
        return 0
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP POST failed: HTTP error {e.code}: {e.reason}")
        raise SystemExit(1) from ValidationError(f"HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        logger.error(f"HTTP POST failed: Failed to reach {args.url}: {e.reason}")
        raise SystemExit(1) from ConfigurationError(f"Failed to reach {args.url}: {e.reason}")
    except TimeoutError as e:
        logger.error(f"HTTP POST failed: Request timeout after 5s: {e}")
        raise SystemExit(1) from ConfigurationError(f"Request timeout after 5s: {e}")
    except (ConnectionError, OSError) as e:
        logger.error(f"HTTP POST failed: Network error during POST request: {e}")
        raise SystemExit(1) from ConfigurationError(f"Network error during POST request: {e}")
    except (ConfigurationError, ValidationError) as e:
        logger.error(f"HTTP POST failed: {e}")
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(130)


if __name__ == '__main__':
    sys.exit(main())
