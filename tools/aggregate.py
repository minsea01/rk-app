#!/usr/bin/env python3
"""Aggregate benchmark results from iperf3 and ffprobe.

This script aggregates network throughput and media metadata into
unified JSON, CSV, and Markdown reports for MCP pipeline validation.
"""

import argparse
import json
import csv
import time
import os
import sys

# Import custom exceptions and logger
from apps.exceptions import ConfigurationError, ValidationError
from apps.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level="INFO")


def read_json(path: str) -> dict:
    """Read and parse JSON file with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON as dictionary

    Raises:
        ConfigurationError: If file cannot be read
        ValidationError: If JSON is invalid
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigurationError(f"Input file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {path}: {e}") from e
    except (IOError, OSError, PermissionError) as e:
        raise ConfigurationError(f"Failed to read {path}: {e}") from e


def frac_to_float(s: str) -> float:
    """Convert fraction string (e.g., '30/1') or float string to float."""
    if not s:
        return 0.0
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            logger.warning(f"Invalid fraction format: '{s}'")
            return 0.0
        try:
            numerator = float(parts[0])
            denominator = float(parts[1])
            if denominator == 0:
                logger.warning(f"Division by zero in fraction: '{s}'")
                return 0.0
            return numerator / denominator
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse fraction '{s}': {e}")
            return 0.0
    try:
        return float(s)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse float '{s}': {e}")
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iperf3", required=True)
    ap.add_argument("--ffprobe", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    logger.info(f"Reading iperf3 results: {args.iperf3}")
    iperf = read_json(args.iperf3)

    logger.info(f"Reading ffprobe results: {args.ffprobe}")
    ffpr = read_json(args.ffprobe)

    # Extract bits_per_second from iperf3 output
    bits_per_second = 0
    try:
        # Prefer received summary
        bits_per_second = iperf["end"]["sum_received"]["bits_per_second"]
    except KeyError:
        try:
            # Fallback to sent summary
            bits_per_second = iperf["end"]["sum_sent"]["bits_per_second"]
        except KeyError:
            # Check if iperf3 encountered an error
            if "error" in iperf:
                logger.warning(f"iperf3 error: {iperf['error']}")
            else:
                logger.warning("No bits_per_second found in iperf3 output")
            bits_per_second = 0

    mbps = bits_per_second / 1e6 if bits_per_second else 0.0

    stream = None
    if "streams" in ffpr and ffpr["streams"]:
        # old ffprobe may print list under 'streams'
        stream = ffpr["streams"][0]
    elif "streams" in ffpr and isinstance(ffpr["streams"], dict) and "streams" in ffpr["streams"]:
        stream = ffpr["streams"]["streams"][0]

    width = int(stream.get("width", 0)) if stream else 0
    height = int(stream.get("height", 0)) if stream else 0
    avg_rate = stream.get("avg_frame_rate", "0/1") if stream else "0/1"
    fps = frac_to_float(avg_rate)
    codec = stream.get("codec_name", "") if stream else ""

    ts = int(time.time())
    summary = {
        "timestamp": ts,
        "iperf3_bits_per_second": bits_per_second,
        "iperf3_mbps": round(mbps, 2),
        "ffprobe_width": width,
        "ffprobe_height": height,
        "ffprobe_fps": round(fps, 2),
        "ffprobe_codec": codec,
    }

    # Write output files with error handling
    try:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:  # dirname returns "" for bare filenames like "report.json"
            os.makedirs(out_dir, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ConfigurationError(f"Failed to create output directory: {e}") from e

    logger.info(f"Writing JSON report: {args.out_json}")
    try:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except (IOError, OSError, PermissionError) as e:
        raise ConfigurationError(f"Failed to write JSON report: {e}") from e

    logger.info(f"Writing CSV report: {args.out_csv}")
    try:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp",
                    "iperf3_mbps",
                    "ffprobe_width",
                    "ffprobe_height",
                    "ffprobe_fps",
                    "ffprobe_codec",
                ]
            )
            w.writerow([ts, round(mbps, 2), width, height, round(fps, 2), codec])
    except (IOError, OSError, PermissionError) as e:
        raise ConfigurationError(f"Failed to write CSV report: {e}") from e

    md = []
    md.append("# Bench Report (MCP MVP)")
    md.append("")
    md.append(f"- Timestamp: {ts}")
    md.append(f"- iperf3 throughput (loopback): {round(mbps,2)} Mbps")
    md.append(f"- Sample video: {width}x{height} @ {round(fps,2)} fps (codec={codec})")
    md.append("")

    logger.info(f"Writing Markdown report: {args.out_md}")
    try:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
    except (IOError, OSError, PermissionError) as e:
        raise ConfigurationError(f"Failed to write Markdown report: {e}") from e

    logger.info("Aggregation completed successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConfigurationError, ValidationError) as e:
        logger.error(f"Aggregation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
