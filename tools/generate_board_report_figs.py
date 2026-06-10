#!/usr/bin/env python3
"""Generate board acceptance figures from reproduced RK3588 measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("artifacts/visualizations")

METRICS = {
    "Network TX\neth1 -> PC": {"value": 942.0, "target": 900.0, "unit": "Mbit/s", "mode": "ge"},
    "Network RX\nPC -> eth1": {"value": 941.0, "target": 900.0, "unit": "Mbit/s", "mode": "ge"},
    "NPU latency\nsingle model": {"value": 18.75, "target": 45.0, "unit": "ms", "mode": "le"},
    "Fake camera\nE2E latency": {"value": 28.924, "target": 45.0, "unit": "ms", "mode": "le"},
    "Fake camera\nFPS": {"value": 33.9623, "target": 30.0, "unit": "FPS", "mode": "ge"},
    "Fake camera\nprocessed": {"value": 1800.0, "target": 1800.0, "unit": "frames", "mode": "ge"},
}


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _passes(value: float, target: float, mode: str) -> bool:
    return value >= target if mode == "ge" else value <= target


def _annotate(ax, bars, labels: list[str]) -> None:
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_target_dashboard() -> Path:
    labels = list(METRICS)
    normalized = []
    value_labels = []
    colors = []

    for label in labels:
        metric = METRICS[label]
        value = metric["value"]
        target = metric["target"]
        mode = metric["mode"]
        unit = metric["unit"]
        score = value / target if mode == "ge" else target / value
        normalized.append(score)
        value_labels.append(f"{value:g} {unit}")
        colors.append("#2a9d8f" if _passes(value, target, mode) else "#c1121f")

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, normalized, color=colors, width=0.62)
    ax.axhline(1.0, color="#264653", linestyle="--", linewidth=1.4, label="Acceptance target")
    ax.set_title("RK3588 Acceptance Target Dashboard")
    ax.set_ylabel("Measured / Target (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(normalized) * 1.2)
    _annotate(ax, bars, value_labels)
    ax.legend(loc="upper left", frameon=True)
    fig.text(
        0.5,
        0.01,
        "Evidence: eth1 iperf3, board_benchmark.py, detect_cli fake-camera 1800-frame TCP run.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.98])

    out = OUT_DIR / "rk3588_acceptance_dashboard.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency_breakout() -> Path:
    labels = ["NPU single model", "C++ fake camera E2E", "Acceptance limit"]
    values = [18.75, 28.924, 45.0]
    colors = ["#457b9d", "#2a9d8f", "#c1121f"]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("RK3588 Latency Evidence")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, 52)
    ax.axhline(45.0, color="#c1121f", linestyle="--", linewidth=1.2)
    _annotate(ax, bars, [f"{v:.2f} ms" for v in values])
    fig.tight_layout()

    out = OUT_DIR / "rk3588_latency_evidence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    for path in [plot_target_dashboard(), plot_latency_breakout()]:
        print(path)


if __name__ == "__main__":
    main()
