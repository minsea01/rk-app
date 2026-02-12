#!/usr/bin/env python3
"""Deprecated wrapper for scripts/profiling/quick_test_optimized.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.deprecation import warn_deprecated


def main() -> int:
    warn_deprecated(
        "apps/quick_test_optimized.py",
        "scripts/profiling/quick_test_optimized.py",
        once=True,
    )
    from scripts.profiling.quick_test_optimized import main as quick_test_main

    return quick_test_main()


if __name__ == "__main__":
    raise SystemExit(main())
