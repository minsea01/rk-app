#!/usr/bin/env python3
import argparse, json, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--file', required=True)
    args = ap.parse_args()
    with open(args.file, 'r', encoding='utf-8') as f:
        payload = f.read().encode('utf-8')
    req = urllib.request.Request(args.url, data=payload, headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=5) as resp:
        print(resp.read().decode('utf-8'))

if __name__ == '__main__':
    main()
