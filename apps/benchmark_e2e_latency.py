#!/usr/bin/env python3
"""Deprecated wrapper for scripts/profiling/benchmark_e2e_latency.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.deprecation import warn_deprecated


def main(argv=None):
    warn_deprecated(
        "apps/benchmark_e2e_latency.py",
        "scripts/profiling/benchmark_e2e_latency.py",
        once=True,
    )
    from scripts.profiling.benchmark_e2e_latency import main as benchmark_main

    if argv is not None:
        return benchmark_main(argv)
    return benchmark_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
