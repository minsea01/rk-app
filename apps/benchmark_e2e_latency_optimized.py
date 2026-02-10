#!/usr/bin/env python3
"""Deprecated wrapper for apps.benchmark_e2e_latency with --optimized."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.deprecation import warn_deprecated
from apps.benchmark_e2e_latency import main as benchmark_main


def _append_optimized_flag(argv: List[str]) -> List[str]:
    if "--optimized" in argv:
        return argv
    return argv + ["--optimized"]


def main() -> int:
    warn_deprecated(
        "apps/benchmark_e2e_latency_optimized.py",
        "apps/benchmark_e2e_latency.py --optimized",
        once=True,
    )
    argv = _append_optimized_flag(sys.argv[1:])
    return benchmark_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

