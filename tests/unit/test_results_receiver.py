#!/usr/bin/env python3
"""Unit tests for scripts.results_receiver."""

import base64
import json

from scripts.results_receiver import ResultsServer


def test_save_result_persists_embedded_jpeg_payload(tmp_path):
    server = ResultsServer(host="127.0.0.1", port=19000, output_dir=str(tmp_path))

    image_bytes = b"\xff\xd8\xff\xd9"
    result = {
        "frame_id": 12,
        "timestamp": 1234567890,
        "detections": [{"class_name": "person", "confidence": 0.95}],
        "image": {
            "encoding": "jpeg",
            "contains_overlays": True,
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        },
    }

    server.save_result(result)

    json_files = sorted(tmp_path.glob("result_*_frame12.json"))
    jpg_files = sorted(tmp_path.glob("result_*_frame12.jpg"))

    assert len(json_files) == 1
    assert len(jpg_files) == 1
    assert jpg_files[0].read_bytes() == image_bytes

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["image"]["encoding"] == "jpeg"
    assert payload["image"]["contains_overlays"] is True
    assert payload["image"]["path"] == jpg_files[0].name
    assert "data_base64" not in payload["image"]


def test_save_result_keeps_json_when_image_payload_missing(tmp_path):
    server = ResultsServer(host="127.0.0.1", port=19001, output_dir=str(tmp_path))

    result = {
        "frame_id": 3,
        "timestamp": 1234567890,
        "detections": [],
        "image": {"encoding": "jpeg"},
    }

    server.save_result(result)

    json_files = sorted(tmp_path.glob("result_*_frame3.json"))
    jpg_files = sorted(tmp_path.glob("result_*_frame3.jpg"))

    assert len(json_files) == 1
    assert not jpg_files

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["image"] == {"encoding": "jpeg"}
