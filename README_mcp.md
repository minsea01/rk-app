MCP Bench MVP for RK3588 Project

This repo includes a minimal, practical MCP-style toolchain to validate the
"build — deploy — observe — archive" loop locally, without real hardware.

What’s included
- configs/mcp_servers.yaml: Declarative list of available tools/servers.
- scripts/run_bench.sh: One-shot runner that executes benchmarks end-to-end.
- tools/iperf3_bench.sh: Starts iperf3 server and runs a local client (loopback).
- tools/ffprobe_probe.sh: Generates a 1080p sample and probes it via ffprobe.
- tools/aggregate.py: Aggregates JSON results into summary JSON/CSV/MD.
- tools/http_receiver.py: Minimal HTTP receiver to accept POSTed JSON.
- tools/http_post.py: Posts the bench summary to the receiver.

Codex MCP servers configured (global ~/.codex/config.toml)
- dev-tools: local development helpers (git diff/summary, search_text, ffprobe, iperf3 client, bench_run, http tools)
- git-summary (alias git): summarize recent git commits in this repo
- github: official GitHub MCP server via npx
  - Requires env: export GITHUB_TOKEN=... (with repo read permissions)
- semgrep: official Semgrep MCP server (python -m semgrep_mcp)
  - Optional env: SEMGREP_APP_TOKEN for SaaS rules; local rules work without
- docker: lightweight Docker MCP (python -m mcp_docker)
  - Uses local docker CLI; read-only tools (info/version/images/ps) plus pull/run
- filesystem: exposes workspace/datasets/artifacts via npx @modelcontextprotocol/server-filesystem
- ssh: optional remote access via npx @idletoaster/ssh-mcp-server (reads ~/.ssh/config)

Quick start
1) Run the full bench pipeline:
   bash scripts/run_bench.sh

2) Results are written to:
   - artifacts/iperf3.json
   - artifacts/ffprobe.json
   - artifacts/bench_summary.json
   - artifacts/bench_summary.csv
   - artifacts/bench_report.md
   - artifacts/http_ingest.log (HTTP receive log)

Notes
- The iperf3 test runs over loopback (127.0.0.1). It validates the toolchain, not NIC hardware.
- ffprobe probes a locally generated 1080p@30fps sample to validate video probing.
- This MVP demonstrates the MCP flow; later swap loopback with real endpoints and camera streams.

Enabling MCP in Codex
- Codex reads config from `~/.codex/config.toml`. This repo added entries under `[mcp_servers.*]` for the servers above.
- Ensure prerequisites:
  - Python 3.10+
  - Node+npx (for GitHub server)
  - `pip install --user semgrep-mcp semgrep`
  - (optional) Docker CLI for docker server
  - (optional) `npm i -g @modelcontextprotocol/server-filesystem @modelcontextprotocol/server-github`
- Set tokens as needed before launching Codex:
  - `export GITHUB_TOKEN=...`
  - (optional) `export SEMGREP_APP_TOKEN=...`
