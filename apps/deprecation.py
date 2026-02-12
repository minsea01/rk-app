#!/usr/bin/env python3
"""Project-wide deprecation helpers."""

from __future__ import annotations

import sys
import warnings
from typing import Set, Tuple

_WARNED: Set[Tuple[str, str, str]] = set()


def format_deprecation_message(old: str, new: str, removal_version: str = "2.0.0") -> str:
    """Return a standard deprecation message."""
    return f"[DEPRECATED] {old} -> {new}, removal in {removal_version}"


def warn_deprecated(
    old: str,
    new: str,
    *,
    removal_version: str = "2.0.0",
    once: bool = True,
) -> str:
    """Emit a standard deprecation warning and stderr log."""
    key = (old, new, removal_version)
    if once and key in _WARNED:
        return format_deprecation_message(old, new, removal_version)

    msg = format_deprecation_message(old, new, removal_version)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    print(msg, file=sys.stderr)

    if once:
        _WARNED.add(key)
    return msg
