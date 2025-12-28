from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

server = FastMCP(name="docker", instructions="Docker helpers via local CLI.")


def _docker_bin() -> str | None:
    return shutil.which("docker")


def _run(subcommand: list[str], timeout: int = 30) -> tuple[str | None, str | None]:
    docker = _docker_bin()
    if not docker:
        return None, "docker CLI not found in PATH"
    proc = subprocess.run(
        [docker, *subcommand],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return None, stderr or stdout or "docker command failed"
    return stdout, None


@server.tool()
def docker_info() -> Any:
    """Return docker info as JSON when possible."""
    out, err = _run(["info", "--format", "{{json .}}"], timeout=25)
    if err:
        return {"error": err}
    try:
        return json.loads(out or "{}")
    except Exception:
        return out or "OK"


@server.tool()
def docker_version() -> Any:
    out, err = _run(["version", "--format", "{{json .}}"], timeout=20)
    if err:
        return {"error": err}
    try:
        return json.loads(out or "{}")
    except Exception:
        return out or "OK"


@server.tool()
def docker_images() -> Any:
    out, err = _run(["images", "--format", "{{json .}}"])
    if err:
        return {"error": err}
    images = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            images.append(json.loads(line))
        except Exception:
            images.append({"raw": line})
    return images


@server.tool()
def docker_ps() -> Any:
    out, err = _run(["ps", "--format", "{{json .}}"])
    if err:
        return {"error": err}
    containers = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            containers.append(json.loads(line))
        except Exception:
            containers.append({"raw": line})
    return containers


@server.tool()
def docker_pull(image: str) -> Any:
    out, err = _run(["pull", image], timeout=120)
    if err:
        return {"error": err}
    return out or "No output."


@server.tool()
def docker_run(image: str, cmd: list[str] | None = None, detach: bool = False) -> Any:
    full_cmd = ["run"]
    if detach:
        full_cmd.append("-d")
    full_cmd.append(image)
    if cmd:
        full_cmd.extend(cmd)

    out, err = _run(full_cmd, timeout=60)
    if err:
        return {"error": err}
    return out or "No output."


if __name__ == "__main__":
    server.run()
