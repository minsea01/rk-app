from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Default workspace; override with WORKDIR if needed.
ROOT = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parent.parent)).resolve()

server = FastMCP(name="dev-tools", instructions="Local dev helpers for rk-app.")


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a command inside the repo and return stdout or raise on failure."""
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    return proc.stdout.strip()


def _safe_path(target: str) -> Path:
    """Resolve a path relative to ROOT and prevent escapes."""
    resolved = (ROOT / target).resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        raise ValueError(f"path {resolved} escapes workdir {ROOT}")
    return resolved


@server.tool()
def summarize_recent_commits(count: int = 10, format: str = "markdown") -> Any:
    """
    Summarize recent git commits.

    Args:
        count: number of commits (1-50).
        format: one of {"markdown", "text", "json"}.
    """
    count = max(1, min(count, 50))
    try:
        raw = _run(
            [
                "git",
                "log",
                f"-{count}",
                "--date=short",
                "--pretty=format:%h|%ad|%an|%s",
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    commits = []
    for line in raw.splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append(
            {
                "hash": parts[0],
                "date": parts[1],
                "author": parts[2],
                "subject": parts[3],
            }
        )

    if format == "json":
        return commits

    lines = [
        f"- {item['hash']} {item['date']} {item['author']}: {item['subject']}"
        for item in commits
    ]
    text = "\n".join(lines) or "No commits found."
    return text if format == "markdown" else text.replace("\n- ", "\n")


@server.tool()
def git_diff_range(
    rev_range: str | None = None, since: str = "HEAD~5", pathspec: str = ""
) -> Any:
    """
    Show a short diffstat for a revision range.

    Args:
        rev_range: explicit git range such as \"v1.0..HEAD\".
        since: used when rev_range is empty (e.g., \"HEAD~10\").
        pathspec: optional path filter.
    """
    target = rev_range.strip() if rev_range else f"{since}..HEAD"
    cmd = ["git", "diff", "--stat", target]
    if pathspec:
        cmd.extend(["--", pathspec])
    try:
        out = _run(cmd, timeout=40)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return out or "No diff."


@server.tool()
def search_text(pattern: str, path: str = ".", max_results: int = 200) -> Any:
    """
    Search text within the repo using ripgrep when available.

    Args:
        pattern: regex or plain text.
        path: subdirectory or file to search.
        max_results: max matches returned (1-500).
    """
    try:
        target = _safe_path(path)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    if not target.exists():
        return {"error": f"{target} not found"}

    limit = max(1, min(max_results, 500))
    rg_bin = shutil.which("rg")
    if rg_bin:
        cmd = [rg_bin, "-n", "--no-heading", "-m", str(limit), pattern, str(target)]
    else:
        cmd = ["grep", "-R", "-n", "-m", str(limit), pattern, str(target)]

    try:
        out = _run(cmd, timeout=40)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    return out or "No matches found."


if __name__ == "__main__":
    server.run()

