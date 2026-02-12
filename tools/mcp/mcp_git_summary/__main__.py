from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

ROOT = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parent.parent)).resolve()
server = FastMCP(name="git-summary", instructions="Git helpers for rk-app.")


def _run(cmd: list[str], timeout: int = 30) -> str:
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


@server.tool()
def summarize_recent_commits(count: int = 10, format: str = "markdown") -> Any:
    """Summarize recent git commits."""
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
        f"- {item['hash']} {item['date']} {item['author']}: {item['subject']}" for item in commits
    ]
    text = "\n".join(lines) or "No commits found."
    return text if format == "markdown" else text.replace("\n- ", "\n")


@server.tool()
def git_diff_range(rev_range: str | None = None, since: str = "HEAD~5", pathspec: str = "") -> Any:
    """Show a short diffstat for a revision range."""
    target = rev_range.strip() if rev_range else f"{since}..HEAD"
    cmd = ["git", "diff", "--stat", target]
    if pathspec:
        cmd.extend(["--", pathspec])
    try:
        out = _run(cmd, timeout=40)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return out or "No diff."


if __name__ == "__main__":
    server.run()
