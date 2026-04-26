#!/usr/bin/env python3
"""Summarize GigE acceptance evidence into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _parse_iperf(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    matches = re.findall(r"\s(\d+(?:\.\d+)?)\s+Mbits/sec\s+\d*\s*sender", text)
    if not matches:
        matches = re.findall(r"\s(\d+(?:\.\d+)?)\s+Mbits/sec\s+receiver", text)
    return {"eth1_mbps": float(matches[-1])} if matches else {}


def _parse_prepare_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    result: dict[str, Any] = {}
    m = re.search(r"PASS=(\d+)\s+WARN=(\d+)\s+FAIL=(\d+)", text)
    if m:
        result["prepare_pass"] = int(m.group(1))
        result["prepare_warn"] = int(m.group(2))
        result["prepare_fail"] = int(m.group(3))
    result["camera_link_detected"] = "[PASS] eth0 link detected" in text
    result["grab_ok"] = "GStreamer grabbed" in text and "[PASS]" in text
    result["detect_ok"] = "detect_cli ran" in text and "[PASS]" in text
    return result


def build_summary(evidence_dir: Path) -> dict[str, Any]:
    tcp_rows = _read_jsonl(evidence_dir / "tcp_sink_results.jsonl")
    board_rows = _read_jsonl(evidence_dir / "board_detect_results.jsonl")
    rows = tcp_rows or board_rows

    summary: dict[str, Any] = {
        "evidence_dir": str(evidence_dir),
        "tcp_frames_received": len(tcp_rows),
        "board_result_rows": len(board_rows),
    }

    if rows:
        first = rows[0]
        last = rows[-1]
        duration_s = (last.get("timestamp", 0) - first.get("timestamp", 0)) / 1000.0
        detections = [d for row in rows for d in row.get("detections", [])]
        summary.update(
            {
                "frames": len(rows),
                "timestamp_duration_s": round(duration_s, 3),
                "result_fps": round(len(rows) / duration_s, 3) if duration_s > 0 else 0,
                "frames_with_detections": sum(1 for row in rows if row.get("detections")),
                "total_detections": len(detections),
                "max_confidence": max(
                    (float(d.get("confidence", 0)) for d in detections), default=0
                ),
            }
        )

    summary.update(_parse_prepare_log(evidence_dir / "gige_acceptance_board.log"))
    summary.update(_parse_iperf(evidence_dir / "iperf_eth1_client.log"))
    return summary


def write_report(evidence_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RK3588 GigE Acceptance Report",
        "",
        f"- Evidence dir: `{evidence_dir}`",
        f"- Prepare checks: PASS={summary.get('prepare_pass', 'n/a')} "
        f"WARN={summary.get('prepare_warn', 'n/a')} FAIL={summary.get('prepare_fail', 'n/a')}",
        f"- Camera link detected: `{summary.get('camera_link_detected', False)}`",
        f"- Grab OK: `{summary.get('grab_ok', False)}`",
        f"- Detect OK: `{summary.get('detect_ok', False)}`",
        f"- TCP frames received: `{summary.get('tcp_frames_received', 0)}`",
        f"- Board JSON rows: `{summary.get('board_result_rows', 0)}`",
        f"- Result FPS: `{summary.get('result_fps', 'n/a')}`",
        f"- eth1 throughput: `{summary.get('eth1_mbps', 'n/a')} Mbps`",
        "",
        "## Notes",
        "",
        "- `eth0` is the camera input NIC.",
        "- `eth1` is the result upload / SSH NIC.",
        "- The generated effective YAML in this evidence folder records the exact camera source URI and thresholds.",
        "",
    ]
    (evidence_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (evidence_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()

    evidence_dir = args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(evidence_dir)
    write_report(evidence_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
