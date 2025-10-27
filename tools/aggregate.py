#!/usr/bin/env python3
import argparse
import json
import csv
import time
import os
import sys
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def frac_to_float(s: str) -> float:
    """Convert fraction string (e.g., '30/1') or float string to float."""
    if not s:
        return 0.0
    if '/' in s:
        parts = s.split('/')
        if len(parts) != 2:
            logging.warning(f"Invalid fraction format: '{s}'")
            return 0.0
        try:
            numerator = float(parts[0])
            denominator = float(parts[1])
            if denominator == 0:
                logging.warning(f"Division by zero in fraction: '{s}'")
                return 0.0
            return numerator / denominator
        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to parse fraction '{s}': {e}")
            return 0.0
    try:
        return float(s)
    except (ValueError, TypeError) as e:
        logging.warning(f"Failed to parse float '{s}': {e}")
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iperf3', required=True)
    ap.add_argument('--ffprobe', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    iperf = read_json(args.iperf3)
    ffpr = read_json(args.ffprobe)

    # Extract bits_per_second from iperf3 output
    bits_per_second = 0
    try:
        # Prefer received summary
        bits_per_second = iperf['end']['sum_received']['bits_per_second']
    except KeyError:
        try:
            # Fallback to sent summary
            bits_per_second = iperf['end']['sum_sent']['bits_per_second']
        except KeyError:
            # Check if iperf3 encountered an error
            if 'error' in iperf:
                logging.warning(f"iperf3 error: {iperf['error']}")
            else:
                logging.warning("No bits_per_second found in iperf3 output")
            bits_per_second = 0

    mbps = bits_per_second / 1e6 if bits_per_second else 0.0

    stream = None
    if 'streams' in ffpr and ffpr['streams']:
        # old ffprobe may print list under 'streams'
        stream = ffpr['streams'][0]
    elif 'streams' in ffpr and isinstance(ffpr['streams'], dict) and 'streams' in ffpr['streams']:
        stream = ffpr['streams']['streams'][0]

    width = int(stream.get('width', 0)) if stream else 0
    height = int(stream.get('height', 0)) if stream else 0
    avg_rate = stream.get('avg_frame_rate', '0/1') if stream else '0/1'
    fps = frac_to_float(avg_rate)
    codec = stream.get('codec_name', '') if stream else ''

    ts = int(time.time())
    summary = {
        'timestamp': ts,
        'iperf3_bits_per_second': bits_per_second,
        'iperf3_mbps': round(mbps, 2),
        'ffprobe_width': width,
        'ffprobe_height': height,
        'ffprobe_fps': round(fps, 2),
        'ffprobe_codec': codec,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','iperf3_mbps','ffprobe_width','ffprobe_height','ffprobe_fps','ffprobe_codec'])
        w.writerow([ts, round(mbps,2), width, height, round(fps,2), codec])

    md = []
    md.append('# Bench Report (MCP MVP)')
    md.append('')
    md.append(f'- Timestamp: {ts}')
    md.append(f'- iperf3 throughput (loopback): {round(mbps,2)} Mbps')
    md.append(f'- Sample video: {width}x{height} @ {round(fps,2)} fps (codec={codec})')
    md.append('')
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

if __name__ == '__main__':
    main()

