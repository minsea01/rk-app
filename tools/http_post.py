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

# Import custom exceptions and logger
from apps.exceptions import ConfigurationError, ValidationError
from apps.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level='INFO')

def main():
    ap = argparse.ArgumentParser(description='POST JSON file to HTTP endpoint')
    ap.add_argument('--url', required=True, help='Target URL')
    ap.add_argument('--file', required=True, help='JSON file to send')
    args = ap.parse_args()

    # Validate input file
    file_path = Path(args.file)
    if not file_path.exists():
        raise ConfigurationError(f"Input file not found: {args.file}")

    logger.info(f"Reading payload from: {args.file}")
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            payload = f.read().encode('utf-8')
    except Exception as e:
        raise ConfigurationError(f"Failed to read payload file: {e}") from e

    # Validate JSON format
    try:
        json.loads(payload.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {args.file}: {e}") from e

    logger.info(f"Sending POST request to: {args.url}")
    req = urllib.request.Request(args.url, data=payload, headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(req, timeout=5) as resp:
            response_text = resp.read().decode('utf-8')
            logger.info(f"Response: {response_text}")
            print(response_text)
        return 0
    except urllib.error.HTTPError as e:
        raise ValidationError(f"HTTP error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise ConfigurationError(f"Failed to reach {args.url}: {e.reason}") from e
    except TimeoutError as e:
        raise ConfigurationError(f"Request timeout after 5s: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to send POST request: {e}") from e

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ConfigurationError, ValidationError) as e:
        logger.error(f"HTTP POST failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
