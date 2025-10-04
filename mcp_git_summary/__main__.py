#!/usr/bin/env python3
import os
import json
import asyncio
import subprocess
from typing import Any, Dict, List

from mcp import types
from mcp.server import Server, NotificationOptions
from mcp import stdio_server


SERVER_NAME = "git-summary"
server = Server(SERVER_NAME, instructions="Summarize recent Git commits in this repo.")


def run_cmd(cmd: List[str], cwd: str | None = None) -> tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def parse_git_log(raw: str) -> List[Dict[str, Any]]:
    commits = []
    for line in raw.strip().splitlines():
        parts = line.split("|", 4)
        if len(parts) < 5:
            # skip unexpected lines
            continue
        short, date, author, refs, subject = parts
        commits.append({
            "hash": short,
            "date": date,
            "author": author,
            "refs": refs.strip(),
            "subject": subject.strip(),
        })
    return commits


@server.list_tools()
async def list_tools(_: types.ListToolsRequest | None = None):
    summarize_input = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "format": {"type": "string", "enum": ["json", "markdown"], "default": "markdown"},
        },
        "additionalProperties": False,
    }
    summarize_output = {
        "type": "object",
        "properties": {
            "commits": {"type": "array"},
        },
        "additionalProperties": True,
    }
    tools = [
        types.Tool(
            name="summarize_recent_commits",
            description="Summarize last N Git commits of the current repository.",
            inputSchema=summarize_input,
            outputSchema=summarize_output,
        )
    ]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    workdir = os.environ.get("WORKDIR", os.getcwd())
    if name == "summarize_recent_commits":
        count = int(arguments.get("count", 10))
        fmt = str(arguments.get("format", "markdown"))
        code, out, err = run_cmd([
            "git", "log", "--graph", "--decorate",
            f"--pretty=format:%h|%ad|%an|%d|%s", "--date=short", f"-n{count}"
        ], cwd=workdir)
        if code != 0:
            text = f"Git log failed: {err.strip()}"
            return [types.TextContent(type="text", text=text)]
        commits = parse_git_log(out)
        if fmt == "json":
            # Return structured content and mirrored text content
            return [types.TextContent(type="text", text=json.dumps({"commits": commits}, ensure_ascii=False, indent=2))], {"commits": commits}
        else:
            lines = ["Recent commits:"]
            for c in commits:
                ref = f" {c['refs']}" if c.get("refs") else ""
                lines.append(f"- {c['hash']} {c['date']} {c['author']}{ref}: {c['subject']}")
            text = "\n".join(lines)
            return [types.TextContent(type="text", text=text)], {"commits": commits}

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def amain():
    # Optionally chdir to WORKDIR for git commands
    workdir = os.environ.get("WORKDIR")
    if workdir:
        try:
            os.chdir(workdir)
        except Exception:
            pass

    init_opts = server.create_initialization_options(notification_options=NotificationOptions())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_opts)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
