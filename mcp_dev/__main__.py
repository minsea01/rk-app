#!/usr/bin/env python3
import os
import re
import json
import shlex
import asyncio
import subprocess
from typing import Any, Dict, List, Tuple

from mcp import types
from mcp.server import Server, NotificationOptions
from mcp import stdio_server


SERVER_NAME = "dev-tools"
server = Server(SERVER_NAME, instructions="Development-oriented MCP tools for this workspace.")


def _workspace_root() -> str:
    # Prefer WORKDIR for clients that pass it in; else current directory
    wd = os.environ.get("WORKDIR") or os.getcwd()
    return os.path.realpath(wd)


def _safe_path(p: str) -> str:
    base = _workspace_root()
    full = os.path.realpath(os.path.join(base, p))
    if not full.startswith(base):
        raise ValueError("path escapes workspace")
    return full


def run_cmd(cmd: List[str], cwd: str | None = None, timeout: int = 20) -> Tuple[int, str, str]:
    try:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError as e:
        return 127, "", str(e)


def parse_git_log(raw: str) -> List[Dict[str, Any]]:
    rows = []
    for line in raw.strip().splitlines():
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        short, date, author, refs, subject = parts
        rows.append({
            "hash": short,
            "date": date,
            "author": author,
            "refs": refs.strip(),
            "subject": subject.strip(),
        })
    return rows


def parse_shortstat(raw: str) -> Dict[str, int]:
    # Example: 12 files changed, 34 insertions(+), 5 deletions(-)
    files = inserts = deletes = 0
    m = re.search(r"(\d+) files? changed", raw)
    if m:
        files = int(m.group(1))
    m = re.search(r"(\d+) insertions?\(\+\)", raw)
    if m:
        inserts = int(m.group(1))
    m = re.search(r"(\d+) deletions?\(-\)", raw)
    if m:
        deletes = int(m.group(1))
    return {"files": files, "insertions": inserts, "deletions": deletes}


@server.list_tools()
async def list_tools(_: types.ListToolsRequest | None = None):
    tools: List[types.Tool] = []

    def tool(name: str, desc: str, input_schema: Dict[str, Any], output_schema: Dict[str, Any] | None = None):
        tools.append(types.Tool(name=name, description=desc, inputSchema=input_schema, outputSchema=output_schema))

    tool(
        "summarize_recent_commits",
        "Summarize last N Git commits.",
        {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"},
            },
            "additionalProperties": False,
        },
        {"type": "object", "properties": {"commits": {"type": "array"}}, "additionalProperties": True},
    )

    tool(
        "git_diff_range",
        "Show diff summary and changed files between revisions.",
        {
            "type": "object",
            "properties": {
                "range": {"type": "string", "description": "e.g. a..b"},
                "since": {"type": "string", "description": "since ref/tag, default HEAD~10"},
                "pathspec": {"type": "string", "description": "optional path filter"},
            },
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "search_text",
        "Search text in workspace (grep -R -n).",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "max_results": {"type": "integer", "default": 200},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        {"type": "object", "properties": {"matches": {"type": "array"}}, "additionalProperties": True},
    )

    tool(
        "ffprobe",
        "Probe media info via ffprobe (JSON).",
        {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "file path or URL"}
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "iperf3_client",
        "Run iperf3 client and return JSON.",
        {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "default": 5201},
                "duration": {"type": "integer", "default": 5},
                "udp": {"type": "boolean", "default": False},
            },
            "required": ["host"],
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "bench_run",
        "Run local bench pipeline scripts/run_bench.sh and collect outputs.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "run_shell",
        "Run a shell command (bash -lc).",
        {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout": {"type": "integer", "default": 30}
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "run_python",
        "Run a short Python snippet in a subprocess.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer", "default": 15}
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "http_get",
        "HTTP GET a URL, return status and body head.",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    tool(
        "http_post_json",
        "HTTP POST JSON to URL, return status and body head.",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}, "json": {"type": "object"}},
            "required": ["url", "json"],
            "additionalProperties": False,
        },
        {"type": "object", "additionalProperties": True},
    )

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    cwd = _workspace_root()

    if name == "summarize_recent_commits":
        count = int(arguments.get("count", 10))
        fmt = str(arguments.get("format", "markdown"))
        code, out, err = run_cmd([
            "git", "log", "--graph", "--decorate",
            f"--pretty=format:%h|%ad|%an|%d|%s", "--date=short", f"-n{count}"
        ], cwd=cwd)
        if code != 0:
            text = f"Git log failed: {err.strip()}"
            return [types.TextContent(type="text", text=text)]
        commits = parse_git_log(out)
        if fmt == "json":
            return [types.TextContent(type="text", text=json.dumps({"commits": commits}, ensure_ascii=False, indent=2))], {"commits": commits}
        lines = ["Recent commits:"]
        for c in commits:
            ref = f" {c['refs']}" if c.get("refs") else ""
            lines.append(f"- {c['hash']} {c['date']} {c['author']}{ref}: {c['subject']}")
        return [types.TextContent(type="text", text="\n".join(lines))], {"commits": commits}

    if name == "git_diff_range":
        rng = arguments.get("range")
        since = arguments.get("since")
        pathspec = arguments.get("pathspec")
        if not rng:
            base = since or "HEAD~10"
            rng = f"{base}..HEAD"
        diff_args = ["git", "diff", "--shortstat", rng]
        diff_args2 = ["git", "diff", "--name-status", rng]
        if pathspec:
            diff_args += ["--", pathspec]
            diff_args2 += ["--", pathspec]
        code1, out1, err1 = run_cmd(diff_args, cwd=cwd)
        code2, out2, err2 = run_cmd(diff_args2, cwd=cwd)
        if code1 != 0 and code2 != 0:
            text = f"git diff failed: {err1 or err2}"
            return [types.TextContent(type="text", text=text)]
        stat = parse_shortstat(out1)
        files = []
        for line in out2.splitlines():
            line = line.strip()
            if not line:
                continue
            status, *rest = line.split("\t")
            path = rest[-1] if rest else ""
            files.append({"status": status, "path": path})
        data = {"range": rng, "shortstat": stat, "files": files}
        text = f"Diff {rng}: {stat['files']} files, +{stat['insertions']}/-{stat['deletions']}"
        return [types.TextContent(type="text", text=text)], data

    if name == "search_text":
        pattern = str(arguments.get("pattern"))
        rel_path = str(arguments.get("path", "."))
        max_results = int(arguments.get("max_results", 200))
        target = _safe_path(rel_path)
        code, out, err = run_cmd(["grep", "-R", "-n", "-I", "-E", pattern, target], cwd=cwd, timeout=25)
        matches = []
        if code in (0, 1):  # 0 found, 1 not found
            for line in out.splitlines():
                if len(matches) >= max_results:
                    break
                # file:line:content
                parts = line.split(":", 2)
                if len(parts) == 3:
                    f, ln, txt = parts
                    matches.append({"file": os.path.relpath(f, cwd), "line": int(ln) if ln.isdigit() else 0, "text": txt})
        data = {"count": len(matches), "matches": matches}
        text = f"Found {len(matches)} matches for /{pattern}/"
        return [types.TextContent(type="text", text=text)], data

    if name == "ffprobe":
        target = str(arguments.get("input"))
        # Allow URL or path; only check path safety if it’s a local path
        if "://" not in target:
            target = _safe_path(target)
        code, out, err = run_cmd(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-print_format", "json", target], cwd=cwd, timeout=20)
        if code != 0:
            return [types.TextContent(type="text", text=f"ffprobe failed: {err.strip()}")]
        try:
            js = json.loads(out)
        except Exception:
            js = {"raw": out}
        return [types.TextContent(type="text", text=json.dumps(js, ensure_ascii=False, indent=2))], js

    if name == "iperf3_client":
        host = str(arguments.get("host"))
        port = int(arguments.get("port", 5201))
        duration = int(arguments.get("duration", 5))
        udp = bool(arguments.get("udp", False))
        args = ["iperf3", "-c", host, "-p", str(port), "-t", str(duration), "-J"]
        if udp:
            args.insert(3, "-u")
        code, out, err = run_cmd(args, cwd=cwd, timeout=duration + 10)
        if code != 0:
            return [types.TextContent(type="text", text=f"iperf3 failed: {err.strip()}")]
        try:
            js = json.loads(out)
        except Exception:
            js = {"raw": out}
        return [types.TextContent(type="text", text=json.dumps(js, ensure_ascii=False, indent=2))], js

    if name == "bench_run":
        script = os.path.join(cwd, "scripts", "run_bench.sh")
        if not os.path.exists(script):
            return [types.TextContent(type="text", text="bench script not found")]
        code, out, err = run_cmd(["bash", script], cwd=cwd, timeout=120)
        # Collect summary files if present
        artifacts_dir = os.path.join(cwd, "artifacts")
        result: Dict[str, Any] = {"exit_code": code, "stdout": out[-4000:], "stderr": err[-2000:]}
        for f in ("bench_summary.json", "bench_report.md", "iperf3.json", "ffprobe.json"):
            p = os.path.join(artifacts_dir, f)
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        if f.endswith(".json"):
                            result[f] = json.load(fh)
                        else:
                            result[f] = fh.read()[-4000:]
                except Exception:
                    pass
        text = f"bench_run exit={code} (see structured output)"
        return [types.TextContent(type="text", text=text)], result

    if name == "http_get":
        import urllib.request
        url = str(arguments.get("url"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(url, timeout=10) as resp:
                body = resp.read(4096).decode("utf-8", errors="ignore")
                data = {"status": resp.status, "headers": dict(resp.headers), "body_head": body}
                return [types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))], data
        except Exception as e:
            return [types.TextContent(type="text", text=f"http_get failed: {e}")]

    if name == "http_post_json":
        import urllib.request
        url = str(arguments.get("url"))
        payload = json.dumps(arguments.get("json", {})).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=10) as resp:
                body = resp.read(4096).decode("utf-8", errors="ignore")
                data = {"status": resp.status, "headers": dict(resp.headers), "body_head": body}
                return [types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))], data
        except Exception as e:
            return [types.TextContent(type="text", text=f"http_post_json failed: {e}")]

    if name == "run_shell":
        cmd = str(arguments.get("cmd"))
        timeout = int(arguments.get("timeout", 30))
        code, out, err = run_cmd(["bash","-lc",cmd], cwd=cwd, timeout=timeout)
        data = {"exit_code": code, "stdout": out[-8000:], "stderr": err[-4000:], "cmd": cmd}
        text = f"run_shell exit={code}"
        return [types.TextContent(type="text", text=text)], data

    if name == "run_python":
        import tempfile, textwrap
        code_text = str(arguments.get("code"))
        timeout = int(arguments.get("timeout", 15))
        # write to temp file to avoid shell quoting issues
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(code_text)
            tmp = tf.name
        try:
            code_rc, out, err = run_cmd(["python3", tmp], cwd=cwd, timeout=timeout)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        data = {"exit_code": code_rc, "stdout": out[-8000:], "stderr": err[-4000:]}
        text = f"run_python exit={code_rc}"
        return [types.TextContent(type="text", text=text)], data

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def amain():
    # Respect WORKDIR if provided by the client
    wd = os.environ.get("WORKDIR")
    if wd:
        try:
            os.chdir(wd)
        except Exception:
            pass
    init_opts = server.create_initialization_options(notification_options=NotificationOptions())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_opts)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
