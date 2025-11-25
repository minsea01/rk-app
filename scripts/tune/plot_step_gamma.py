#!/usr/bin/env python3
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BENCH = os.path.join(ROOT, 'build', 'x86-debug', 'pid_bench')

def ensure_bench():
    if not os.path.exists(BENCH):
        print('bench binary not found. Build first: cmake --preset x86-debug && cmake --build --preset x86-debug -j', file=sys.stderr)
        sys.exit(2)

def run_curve(gamma, csv_path, kp, ki, kd, umin, umax, steps, dt, sp, plant='second'):
    env = os.environ.copy()
    env.setdefault('ASAN_OPTIONS', 'detect_leaks=0')
    cmd = [BENCH,
           '--impl=c', f'--plant={plant}', f'--dt={dt}', f'--steps={steps}', f'--sp={sp}',
           f'--kp={kp}', f'--ki={ki}', f'--kd={kd}', f'--umin={umin}', f'--umax={umax}',
           '--beta=1.0', f'--gamma={gamma}', '--kdN=20.0', f'--csv={csv_path}']
    subprocess.check_call(cmd, env=env)

def plot_png(csv_files, labels, out_png, ref_value=1.0):
    try:
        import matplotlib.pyplot as plt
        import csv
    except ImportError:
        print('matplotlib not available. Please: pip install matplotlib', file=sys.stderr)
        print('CSV files are generated:')
        for p in csv_files:
            print(' -', p)
        return

    plt.figure(figsize=(8, 4.5))
    t_ref = None
    for csv_path, label in zip(csv_files, labels):
        t, y = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t.append(float(row['t']))
                y.append(float(row['y']))
        plt.plot(t, y, label=label)
        if t_ref is None:
            t_ref = t
    if t_ref is not None:
        plt.plot(t_ref, [ref_value]*len(t_ref), linestyle='--', color='k', linewidth=1.0, label=f'ref={ref_value}')
    plt.xlabel('Time (s)')
    plt.ylabel('Output y')
    plt.title('PID Step Response vs gamma (2DOF, C impl)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print('Plot saved to', out_png)

def main():
    ensure_bench()
    out_dir = os.path.join(ROOT, 'results', 'tune')
    os.makedirs(out_dir, exist_ok=True)

    dt = 0.001
    steps = 20000
    sp = 1.0
    kp, ki, kd = 2.0, 1.0, 0.1
    umin, umax = -500, 500
    gammas = [0.0, 0.5, 1.0]

    csvs = []
    labels = []
    for g in gammas:
        csv_path = os.path.join(out_dir, f'step_gamma_{g:.1f}.csv')
        run_curve(g, csv_path, kp, ki, kd, umin, umax, steps, dt, sp)
        csvs.append(csv_path)
        labels.append(f'gamma={g:.1f}')

    plot_png(csvs, labels, os.path.join(out_dir, 'step_gamma.png'), ref_value=sp)

if __name__ == '__main__':
    main()
