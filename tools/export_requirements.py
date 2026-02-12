#!/usr/bin/env python3
"""Generate requirements files from pyproject.toml."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQ_BASE = REPO_ROOT / "requirements.txt"
REQ_DEV = REPO_ROOT / "requirements-dev.txt"
REQ_BOARD = REPO_ROOT / "requirements_board.txt"
REQ_TRAIN = REPO_ROOT / "requirements_train.txt"

PACKAGE_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def requirement_key(spec: str) -> str:
    match = PACKAGE_RE.match(spec)
    if not match:
        return spec.strip().lower()
    return match.group(1).lower()


def merge_requirements(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    index_by_key: dict[str, int] = {}
    for group in groups:
        for raw in group:
            spec = raw.strip()
            if not spec:
                continue
            key = requirement_key(spec)
            if key in index_by_key:
                merged[index_by_key[key]] = spec
            else:
                index_by_key[key] = len(merged)
                merged.append(spec)
    return merged


def render_requirements(title: str, requirements: list[str]) -> str:
    lines = [
        f"# Auto-generated from pyproject.toml: {title}",
        "# Do not edit manually. Run: python3 tools/export_requirements.py",
        "",
    ]
    lines.extend(requirements)
    lines.append("")
    return "\n".join(lines)


def load_dependencies() -> tuple[list[str], list[str], list[str], list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional = project.get("optional-dependencies", {})
    base = [str(x) for x in project.get("dependencies", [])]
    dev = [str(x) for x in optional.get("dev", [])]
    board = [str(x) for x in optional.get("board", [])]
    train = [str(x) for x in optional.get("train", [])]
    return base, dev, board, train


def expected_outputs() -> dict[Path, str]:
    base, dev, board, train = load_dependencies()
    base_merged = merge_requirements(base)
    dev_merged = merge_requirements(base, dev)
    board_merged = merge_requirements(base, board)
    train_merged = merge_requirements(base, train)
    return {
        REQ_BASE: render_requirements("base", base_merged),
        REQ_DEV: render_requirements("base+dev", dev_merged),
        REQ_BOARD: render_requirements("base+board", board_merged),
        REQ_TRAIN: render_requirements("base+train", train_merged),
    }


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"updated {path.relative_to(REPO_ROOT)}")


def check_outputs(outputs: dict[Path, str]) -> int:
    mismatches: list[str] = []
    for path, expected in outputs.items():
        if not path.exists():
            mismatches.append(f"missing: {path.relative_to(REPO_ROOT)}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            mismatches.append(f"outdated: {path.relative_to(REPO_ROOT)}")
    if mismatches:
        print("requirements export check failed:")
        for item in mismatches:
            print(f"  - {item}")
        print("run: python3 tools/export_requirements.py")
        return 1
    print("requirements export check passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check files are up-to-date")
    args = parser.parse_args()

    outputs = expected_outputs()
    if args.check:
        return check_outputs(outputs)

    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
