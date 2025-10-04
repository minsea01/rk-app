#!/usr/bin/env python3
import argparse, json, csv, time, os, sys

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def frac_to_float(s: str) -> float:
    if not s:
        return 0.0
    if '/' in s:
        n, d = s.split('/')
        try:
            return float(n) / float(d)
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
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

    bits_per_second = None
    try:
        # prefer received summary
        bits_per_second = iperf['end']['sum_received']['bits_per_second']
    except Exception:
        try:
            bits_per_second = iperf['end']['sum_sent']['bits_per_second']
        except Exception:
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

