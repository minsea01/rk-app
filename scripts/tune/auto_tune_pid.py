#!/usr/bin/env python3
"""
Auto-tune PID (grid or random search) using the pid_bench binary.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BENCH = os.path.join(ROOT, 'build', 'x86-debug', 'pid_bench')
OUTROOT = os.path.join(ROOT, 'results', 'autotune')


def ensure_bench():
    if not os.path.exists(BENCH):
        print('bench binary not found. Build first: cmake --preset x86-debug && cmake --build --preset x86-debug -j', file=sys.stderr)
        sys.exit(2)


def run_bench(args_list):
    env = os.environ.copy()
    env.setdefault('ASAN_OPTIONS', 'detect_leaks=0')
    try:
        out = subprocess.check_output([BENCH] + args_list, env=env, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        return None, f"bench failed: {e.output.strip()}"
    metrics = {}
    for line in out.splitlines():
        if line.strip().startswith('IAE='):
            parts = [p.strip() for p in line.split(',')]
            for p in parts:
                if p.startswith('IAE='): metrics['IAE'] = float(p.split('=')[1])
                elif p.startswith('ISE='): metrics['ISE'] = float(p.split('=')[1])
                elif p.startswith('rise_t='):
                    val = p.split('=')[1].split()[0]
                    metrics['rise_t'] = float(val) if val != 'nan' else math.nan
                elif p.startswith('settle_t'):
                    val = p.split('~')[1].split()[0]
                    metrics['settle_t'] = float(val) if val != 'nan' else math.nan
                elif p.startswith('overshoot='):
                    metrics['overshoot'] = float(p.split('=')[1].split()[0])
            return metrics, None
    return None, 'metrics not found'


@dataclass
class Caps:
    rise_cap: float = 5.0
    settle_cap: float = 10.0
    over_cap: float = 100.0


@dataclass
class Weights:
    w_iae: float = 1.0
    w_ise: float = 0.0
    w_rise: float = 0.1
    w_settle: float = 0.05
    w_over: float = 0.01
    w_rise_target: float = 0.5
    w_settle_target: float = 0.2


@dataclass
class Constraints:
    max_overshoot: float | None = None
    max_settle: float | None = None

@dataclass
class Targets:
    target_rise: float | None = None
    target_settle: float | None = None
    only_penalize_slower: bool = True

def cost_of(metrics, caps, weights, constraints, targets):
    iae = metrics.get('IAE', math.inf)
    ise = metrics.get('ISE', math.inf)
    rise = metrics.get('rise_t', math.nan)
    settle = metrics.get('settle_t', math.nan)
    over = metrics.get('overshoot', math.inf)

    if constraints.max_overshoot is not None and over > constraints.max_overshoot:
        return math.inf
    if constraints.max_settle is not None:
        if math.isnan(settle) or settle > constraints.max_settle:
            return math.inf

    if math.isnan(rise):
        rise = caps.rise_cap
    if math.isnan(settle):
        settle = caps.settle_cap

    rise = min(rise, caps.rise_cap)
    settle = min(settle, caps.settle_cap)
    over = min(over, caps.over_cap)

    pr = 0.0
    ps = 0.0
    if targets.target_rise is not None:
        dr = max(0.0, rise - targets.target_rise) if targets.only_penalize_slower else abs(rise - targets.target_rise)
        pr = weights.w_rise_target * dr
    if targets.target_settle is not None:
        ds = max(0.0, settle - targets.target_settle) if targets.only_penalize_slower else abs(settle - targets.target_settle)
        ps = weights.w_settle_target * ds

    return (
        weights.w_iae * iae +
        weights.w_ise * ise +
        weights.w_rise * rise +
        weights.w_settle * settle +
        weights.w_over * over +
        pr + ps
    )


def build_args(ns, kp, ki, kd, csv_path=None):
    args = [
        f"--impl={ns.impl}", f"--plant={ns.plant}", f"--dt={ns.dt}", f"--steps={ns.steps}", f"--sp={ns.sp}",
        f"--umin={ns.umin}", f"--umax={ns.umax}", f"--kp={kp}", f"--ki={ki}", f"--kd={kd}",
    ]
    if getattr(ns, 'settle_hold', None) is not None:
        args += [f"--settle_hold={ns.settle_hold}"]
    if ns.plant == 'first':
        args += [f"--tau={ns.tau}", f"--K={ns.K}"]
    else:
        args += [f"--zeta={ns.zeta}", f"--wn={ns.wn}", f"--K={ns.K}"]
    if ns.impl == 'c':
        args += [f"--beta={ns.beta}", f"--gamma={ns.gamma}", f"--kdN={ns.kdN}"]
        if ns.kff is not None:
            args += [f"--kff={ns.kff}"]
        if ns.ffk0 is not None:
            args += [f"--ffk0={ns.ffk0}"]
        if ns.ffk1 is not None:
            args += [f"--ffk1={ns.ffk1}"]
        if ns.ffk2 is not None:
            args += [f"--ffk2={ns.ffk2}"]
        if ns.spf:
            args += ["--spf"]
            if ns.spf_wn is not None:
                args += [f"--spf-wn={ns.spf_wn}"]
            if ns.spf_zeta is not None:
                args += [f"--spf-zeta={ns.spf_zeta}"]
    if csv_path:
        args += [f"--csv={csv_path}"]
    return args


def main():
    p = argparse.ArgumentParser(description='Auto-tune PID via grid/random search running pid_bench')
    p.add_argument('--profile', choices=['none','fast100','lowovershoot'], default='none',
                   help='preconfigured search: fast100 (~0.11s rise), lowovershoot (<1% overshoot)')
    p.add_argument('--impl', choices=['cpp','c'], default='cpp')
    p.add_argument('--plant', choices=['first','second'], default='second')
    p.add_argument('--dt', type=float, default=0.001)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--sp', type=float, default=1.0)
    p.add_argument('--settle-hold', dest='settle_hold', type=float, default=None, help='hold duration (s) for settle detection in bench')
    p.add_argument('--umin', type=float, default=-500)
    p.add_argument('--umax', type=float, default=500)
    p.add_argument('--tau', type=float, default=0.05)
    p.add_argument('--K', type=float, default=1.0)
    p.add_argument('--zeta', type=float, default=0.7)
    p.add_argument('--wn', type=float, default=20.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--kdN', type=float, default=20.0)
    p.add_argument('--kff', type=float, default=None)
    p.add_argument('--ffk0', type=float, default=None)
    p.add_argument('--ffk1', type=float, default=None)
    p.add_argument('--ffk2', type=float, default=None)
    p.add_argument('--spf', action='store_true')
    p.add_argument('--spf-wn', dest='spf_wn', type=float, default=None)
    p.add_argument('--spf-zeta', dest='spf_zeta', type=float, default=None)
    p.add_argument('--mode', choices=['grid','random'], default='grid')
    p.add_argument('--kp-min', type=float, default=0.5)
    p.add_argument('--kp-max', type=float, default=6.0)
    p.add_argument('--kp-steps', type=int, default=12)
    p.add_argument('--ki-min', type=float, default=0.0)
    p.add_argument('--ki-max', type=float, default=3.0)
    p.add_argument('--ki-steps', type=int, default=13)
    p.add_argument('--kd-min', type=float, default=0.0)
    p.add_argument('--kd-max', type=float, default=0.30)
    p.add_argument('--kd-steps', type=int, default=7)
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-overshoot', type=float, default=None)
    p.add_argument('--max-settle', type=float, default=None)
    p.add_argument('--target-rise', type=float, default=None)
    p.add_argument('--target-settle', type=float, default=None)
    p.add_argument('--only-penalize-slower', action='store_true')
    p.add_argument('--w-iae', type=float, default=1.0)
    p.add_argument('--w-ise', type=float, default=0.0)
    p.add_argument('--w-rise', type=float, default=0.1)
    p.add_argument('--w-settle', type=float, default=0.05)
    p.add_argument('--w-over', type=float, default=0.01)
    p.add_argument('--w-rise-target', type=float, default=0.5)
    p.add_argument('--w-settle-target', type=float, default=0.2)
    p.add_argument('--outdir', type=str, default=OUTROOT, help='directory to write results/plots (default: results/autotune)')
    p.add_argument('--top', type=int, default=10, help='print/save top-N candidates')
    p.add_argument('--out', type=str, default=None, help='results JSON filepath (defaults to <outdir>/autotune_results.json)')
    p.add_argument('--plot-best', action='store_true')
    p.add_argument('--ff-auto', action='store_true')
    p.add_argument('--ff-dyn-auto', action='store_true')
    p.add_argument('--refine', action='store_true')
    p.add_argument('--refine-steps', type=int, default=7)
    p.add_argument('--refine-span', type=float, default=0.5)

    ns = p.parse_args()
    ensure_bench()
    outdir = ns.outdir
    os.makedirs(outdir, exist_ok=True)

    caps = Caps()
    weights = Weights(ns.w_iae, ns.w_ise, ns.w_rise, ns.w_settle, ns.w_over, ns.w_rise_target, ns.w_settle_target)
    constraints = Constraints(ns.max_overshoot, ns.max_settle)
    targets = Targets(ns.target_rise, ns.target_settle, ns.only_penalize_slower)

    # Profiles
    if ns.profile != 'none':
        ns.impl = 'c'
        ns.plant = 'second'
        ns.dt = 0.001
        ns.steps = max(ns.steps, 10000)
        ns.kdN = 100.0
        ns.spf = True
        if ns.spf_wn is None:
            ns.spf_wn = 40.0
        if ns.spf_zeta is None:
            ns.spf_zeta = 1.0
        ns.ff_auto = True
        ns.ff_dyn_auto = True
        ns.ki_min, ns.ki_max, ns.ki_steps = 0.0, 0.0, 1
        ns.mode = 'grid'
        ns.umin = min(ns.umin, -500.0)
        ns.umax = max(ns.umax,  500.0)
        if ns.profile == 'fast100':
            ns.target_rise = 0.11
            ns.only_penalize_slower = True
            ns.max_overshoot = 8.0
            ns.w_rise_target = max(ns.w_rise_target, 1.5)
            ns.w_over = max(ns.w_over, 0.2)
            ns.kp_min, ns.kp_max, ns.kp_steps = 0.3, 1.4, 12
            ns.kd_min, ns.kd_max, ns.kd_steps = 0.6, 2.0, 10
            ns.refine = True
            ns.refine_steps = max(ns.refine_steps, 9)
            ns.refine_span = max(ns.refine_span, 0.6)
        elif ns.profile == 'lowovershoot':
            ns.max_overshoot = 1.0
            ns.target_rise = 0.14
            ns.only_penalize_slower = True
            ns.w_over = max(ns.w_over, 2.0)
            ns.w_rise_target = max(ns.w_rise_target, 1.0)
            ns.kp_min, ns.kp_max, ns.kp_steps = 0.2, 1.2, 8
            ns.kd_min, ns.kd_max, ns.kd_steps = 0.5, 1.6, 12
            ns.refine = True
            ns.refine_steps = max(ns.refine_steps, 9)
            ns.refine_span = max(ns.refine_span, 0.6)

    random.seed(ns.seed)
    results = []

    # auto feedforward
    if ns.impl == 'c' and ns.plant == 'second' and ns.ff_auto:
        ns.ffk0 = (ns.wn * ns.wn) / ns.K
        if ns.ff_dyn_auto:
            ns.ffk1 = (2.0 * ns.zeta * ns.wn) / ns.K
            ns.ffk2 = 1.0 / ns.K

    def record(kp, ki, kd):
        metrics, err = run_bench(build_args(ns, kp, ki, kd))
        if metrics is None:
            return
        c = (
            cost_of(metrics, caps, weights, constraints, targets)
        )
        if math.isfinite(c):
            results.append({'kp': kp, 'ki': ki, 'kd': kd, 'metrics': metrics, 'cost': c})

    if ns.mode == 'grid':
        def linspace(a, b, n):
            if n <= 1: return [a]
            step = (b - a) / (n - 1)
            return [a + i * step for i in range(n)]
        for kp in linspace(ns.kp_min, ns.kp_max, ns.kp_steps):
            for ki in linspace(ns.ki_min, ns.ki_max, ns.ki_steps):
                for kd in linspace(ns.kd_min, ns.kd_max, ns.kd_steps):
                    record(kp, ki, kd)
    else:
        for _ in range(ns.samples):
            kp = random.uniform(ns.kp_min, ns.kp_max)
            ki = random.uniform(ns.ki_min, ns.ki_max)
            kd = random.uniform(ns.kd_min, ns.kd_max)
            record(kp, ki, kd)

    if not results:
        print('No valid results (constraints too strict?)', file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda r: r['cost'])
    best = results[0]

    if ns.refine:
        base_kp, base_ki, base_kd = best['kp'], best['ki'], best['kd']
        span_kp = (ns.kp_max - ns.kp_min) / max(1, ns.kp_steps - 1)
        span_ki = (ns.ki_max - ns.ki_min) / max(1, ns.ki_steps - 1)
        span_kd = (ns.kd_max - ns.kd_min) / max(1, ns.kd_steps - 1)
        span_kp *= ns.refine_span
        span_ki *= ns.refine_span
        span_kd *= ns.refine_span
        fine = []
        def lin(a,b,n):
            if n<=1: return [a]
            step=(b-a)/(n-1); return [a+i*step for i in range(n)]
        for kp in lin(base_kp - span_kp, base_kp + span_kp, ns.refine_steps):
            for ki in lin(base_ki - span_ki, base_ki + span_ki, ns.refine_steps):
                for kd in lin(base_kd - span_kd, base_kd + span_kd, ns.refine_steps):
                    metrics, err = run_bench(build_args(ns, kp, ki, kd))
                    if metrics is None: continue
                    c = cost_of(metrics, caps, weights, constraints, targets)
                    if math.isfinite(c):
                        fine.append({'kp': kp, 'ki': ki, 'kd': kd, 'metrics': metrics, 'cost': c})
        if fine:
            fine.sort(key=lambda r: r['cost'])
            results = fine + results
            best = fine[0]

    # Resolve paths
    out_json = ns.out if ns.out else os.path.join(outdir, 'autotune_results.json')
    alias_out_dir = os.path.join(ROOT, 'out')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # Save JSON
    with open(out_json, 'w') as f:
        json.dump({'args': vars(ns), 'top': results[:ns.top]}, f, indent=2)
    print(f"Saved results to {out_json}")

    # Print top
    print('Top candidates:')
    for i, r in enumerate(results[:ns.top], start=1):
        m = r['metrics']
        print(f"{i:2d}) kp={r['kp']:.3f} ki={r['ki']:.3f} kd={r['kd']:.3f} | cost={r['cost']:.3f} "
              f"IAE={m['IAE']:.3f} ISE={m['ISE']:.3f} rise={m.get('rise_t', float('nan'))} "
              f"settle={m.get('settle_t', float('nan'))} over={m.get('overshoot', 0.0)}%")

    # Plot best
    if ns.plot_best:
        csv_path = os.path.join(outdir, 'autotune_best.csv')
        run_bench(build_args(ns, best['kp'], best['ki'], best['kd'], csv_path=csv_path))
        try:
            import matplotlib.pyplot as plt
            import csv
            t, y = [], []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t.append(float(row['t']))
                    y.append(float(row['y']))
            os.makedirs(outdir, exist_ok=True)
            plt.figure(figsize=(8, 4.5))
            plt.plot(t, y, label='y')
            plt.plot(t, [ns.sp]*len(t), linestyle='--', color='k', linewidth=1.0, label='ref')
            plt.xlabel('Time (s)'); plt.ylabel('Output y'); plt.grid(True); plt.legend()
            png = os.path.join(outdir, 'autotune_best.png')
            plt.tight_layout(); plt.savefig(png, dpi=150)
            print('Best plot saved to', png)
        except Exception:
            print('matplotlib not available; CSV saved to', csv_path)

    # Back-compat alias: also copy key files to out/
    try:
        import shutil
        os.makedirs(alias_out_dir, exist_ok=True)
        # Copy JSON
        shutil.copy2(out_json, os.path.join(alias_out_dir, 'autotune_results.json'))
        # Copy CSV/PNG if exist
        for name in ('autotune_best.csv', 'autotune_best.png'):
            p = os.path.join(outdir, name)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(alias_out_dir, name))
    except Exception:
        pass


if __name__ == '__main__':
    main()
