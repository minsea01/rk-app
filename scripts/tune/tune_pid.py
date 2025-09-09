#!/usr/bin/env python3
import itertools
import math
import os
import re
import subprocess
import sys

BENCH = os.path.join("build", "x86-debug", "pid_bench")

def run_once(args):
    env = os.environ.copy()
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    cmd = [BENCH] + args
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        return None, f"run failed: {e.output.strip()}"
    m = {}
    for line in out.splitlines():
        if "IAE=" in line:
            rec = dict()
            try:
                for part in line.strip().split(','):
                    if 'IAE=' in part: rec['IAE'] = float(part.split('=')[1])
                    elif 'ISE=' in part: rec['ISE'] = float(part.split('=')[1])
                    elif 'rise_t=' in part: rec['rise_t'] = float(part.split('=')[1].split()[0]) if 'nan' not in part else math.nan
                    elif 'settle_t' in part:
                        val = part.split('~')[1].split()[0]
                        rec['settle_t'] = float(val) if val != 'nan' else math.nan
                    elif 'overshoot=' in part:
                        rec['overshoot'] = float(part.split('=')[1].split()[0])
                return rec, None
            except Exception as ex:
                return None, f"parse error: {ex}; line={line}"
    return None, "no metrics line found"

def main():
    if not os.path.exists(BENCH):
        print("bench binary not found; build x86-debug first", file=sys.stderr)
        sys.exit(2)

    grid = {
        'kp': [0.8, 1.2, 1.5, 2.0, 2.5],
        'ki': [0.2, 0.4, 0.6, 0.8, 1.0],
        'kd': [0.00, 0.02, 0.05, 0.08, 0.12],
    }

    results = []
    for kp, ki, kd in itertools.product(grid['kp'], grid['ki'], grid['kd']):
        args = [
            "--impl=cpp", "--plant=second", "--dt=0.001", "--steps=20000",
            f"--kp={kp}", f"--ki={ki}", f"--kd={kd}", "--umin=-500", "--umax=500",
        ]
        rec, err = run_once(args)
        if rec is None:
            continue
        rise = rec.get('rise_t', math.inf)
        settle = rec.get('settle_t', math.inf)
        over = rec.get('overshoot', 0.0)
        if math.isnan(rise): rise = 1e3
        if math.isnan(settle): settle = 1e3
        cost = rec['IAE'] + 0.01*over + 0.1*min(rise, 5.0) + 0.05*min(settle, 10.0)
        results.append((cost, kp, ki, kd, rec))

    results.sort(key=lambda x: x[0])
    print("Top 10 (by cost):")
    for i, (cost, kp, ki, kd, rec) in enumerate(results[:10], 1):
        print(f"{i:2d}) kp={kp:.3f} ki={ki:.3f} kd={kd:.3f} | cost={cost:.3f} IAE={rec['IAE']:.3f} ISE={rec['ISE']:.3f} rise={rec.get('rise_t', float('nan'))} settle={rec.get('settle_t', float('nan'))} over={rec.get('overshoot', 0.0)}%")

if __name__ == '__main__':
    main()
