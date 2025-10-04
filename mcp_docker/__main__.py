#!/usr/bin/env python3
import os, json, asyncio, subprocess
from typing import Any, Dict, List, Tuple
from mcp import types
from mcp.server import Server, NotificationOptions
from mcp import stdio_server

server = Server("docker-tools", instructions="Docker utilities via MCP (info/ps/images/pull/run)")


def run_cmd(args: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


@server.list_tools()
async def list_tools(_: types.ListToolsRequest | None = None):
    tools: List[types.Tool] = []
    def tool(name: str, desc: str, input_schema: Dict[str, Any] | None = None):
        tools.append(
            types.Tool(
                name=name,
                description=desc,
                inputSchema=input_schema or {"type": "object", "properties": {}, "additionalProperties": False},
                outputSchema={"type":"object","additionalProperties":True},
            )
        )
    tool("docker_info", "docker info in JSON if available")
    tool("docker_version", "docker version in JSON if available")
    tool("docker_images", "list images", {"type":"object","properties":{},"additionalProperties":False})
    tool("docker_ps", "list containers", {"type":"object","properties":{},"additionalProperties":False})
    tool("docker_pull", "pull an image", {"type":"object","properties":{"image":{"type":"string"}},"required":["image"],"additionalProperties":False})
    tool("docker_run", "run a container (simple)", {"type":"object","properties":{
        "image":{"type":"string"},
        "cmd":{"type":"array","items":{"type":"string"},"default":[]},
        "detach":{"type":"boolean","default":False}
    },"required":["image"],"additionalProperties":False})
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name == "docker_info":
        code, out, err = run_cmd(["docker","info","--format","{{json .}}"], 20)
        if code != 0:
            return [types.TextContent(type="text", text=f"docker info failed: {err}")]
        try:
            js = json.loads(out)
        except Exception:
            js = {"raw": out}
        return [types.TextContent(type="text", text=json.dumps(js, ensure_ascii=False, indent=2))], js

    if name == "docker_version":
        code, out, err = run_cmd(["docker","version","--format","{{json .}}"], 15)
        if code != 0:
            return [types.TextContent(type="text", text=f"docker version failed: {err}")]
        try:
            js = json.loads(out)
        except Exception:
            js = {"raw": out}
        return [types.TextContent(type="text", text=json.dumps(js, ensure_ascii=False, indent=2))], js

    if name == "docker_images":
        code, out, err = run_cmd(["docker","images","--format","{{json .}}"], 20)
        if code != 0:
            return [types.TextContent(type="text", text=f"docker images failed: {err}")]
        lines = [json.loads(l) for l in out.splitlines() if l.strip()]
        return [types.TextContent(type="text", text=json.dumps(lines, ensure_ascii=False, indent=2))], {"images": lines}

    if name == "docker_ps":
        code, out, err = run_cmd(["docker","ps","--format","{{json .}}"], 20)
        if code != 0:
            return [types.TextContent(type="text", text=f"docker ps failed: {err}")]
        lines = [json.loads(l) for l in out.splitlines() if l.strip()]
        return [types.TextContent(type="text", text=json.dumps(lines, ensure_ascii=False, indent=2))], {"containers": lines}

    if name == "docker_pull":
        image = str(arguments.get("image"))
        code, out, err = run_cmd(["docker","pull", image], 300)
        data = {"exit_code": code, "stdout": out[-4000:], "stderr": err[-2000:]}
        text = f"docker pull {image} exit={code}"
        return [types.TextContent(type="text", text=text)], data

    if name == "docker_run":
        image = str(arguments.get("image"))
        cmd = list(arguments.get("cmd", []))
        detach = bool(arguments.get("detach", False))
        argv = ["docker","run"] + (["-d"] if detach else []) + [image] + cmd
        code, out, err = run_cmd(argv, 120)
        data = {"exit_code": code, "stdout": out[-4000:], "stderr": err[-2000:], "argv": argv}
        text = f"docker run {image} exit={code}"
        return [types.TextContent(type="text", text=text)], data

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def amain():
    init_opts = server.create_initialization_options(notification_options=NotificationOptions())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_opts)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
