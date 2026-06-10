#!/usr/bin/env python3
"""Unit tests for scripts.live_web_viewer."""

import base64

from scripts.live_web_viewer import LiveFrameState, decode_embedded_jpeg, sanitize_result


def test_decode_embedded_jpeg_accepts_base64_payload():
    image_bytes = b"\xff\xd8\xff\xd9"
    result = {
        "image": {
            "encoding": "jpeg",
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        }
    }

    assert decode_embedded_jpeg(result) == image_bytes


def test_sanitize_result_removes_large_image_payload():
    result = {
        "frame_id": 7,
        "detections": [{"class_name": "person", "confidence": 0.9}],
        "image": {"encoding": "jpeg", "data_base64": "large"},
    }

    sanitized = sanitize_result(result)

    assert sanitized["image"] == {"encoding": "jpeg"}
    assert result["image"]["data_base64"] == "large"


def test_live_frame_state_tracks_latest_jpeg_and_metadata(tmp_path):
    image_bytes = b"\xff\xd8\xff\xd9"
    result = {
        "frame_id": 42,
        "timestamp": 123.4,
        "detections": [{"class_name": "person", "confidence": 0.95}],
        "image": {
            "encoding": "jpeg",
            "contains_overlays": True,
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        },
    }
    state = LiveFrameState(
        save_latest=tmp_path / "latest.jpg",
        save_dir=tmp_path / "frames",
    )

    state.update_from_result(result, raw_size=123)

    snapshot = state.snapshot()
    assert snapshot["has_image"] is True
    assert snapshot["latest_frame_id"] == 42
    assert snapshot["latest_result"]["image"] == {
        "encoding": "jpeg",
        "contains_overlays": True,
    }
    assert snapshot["stats"]["results"] == 1
    assert snapshot["stats"]["images"] == 1
    assert snapshot["stats"]["bytes_received"] == 123
    assert (tmp_path / "latest.jpg").read_bytes() == image_bytes
    assert (tmp_path / "frames" / "frame_000042.jpg").read_bytes() == image_bytes


def test_live_frame_state_keeps_latest_image_path_across_result_only_frames(tmp_path):
    image_bytes = b"\xff\xd8\xff\xd9"
    state = LiveFrameState(save_latest=tmp_path / "latest.jpg", save_dir=None)
    state.update_from_result(
        {
            "frame_id": 1,
            "image": {
                "encoding": "jpeg",
                "data_base64": base64.b64encode(image_bytes).decode("ascii"),
            },
        },
        raw_size=10,
    )

    state.update_from_result({"frame_id": 2, "detections": []}, raw_size=5)

    snapshot = state.snapshot()
    assert snapshot["latest_frame_id"] == 2
    assert snapshot["latest_saved_path"] == str(tmp_path / "latest.jpg")
    assert snapshot["stats"]["results"] == 2
    assert snapshot["stats"]["images"] == 1


def test_live_frame_state_prunes_saved_frame_directory(tmp_path):
    state = LiveFrameState(
        save_latest=tmp_path / "latest.jpg",
        save_dir=tmp_path / "frames",
        save_dir_max_files=2,
    )

    for frame_id in range(3):
        state.update_from_result(
            {
                "frame_id": frame_id,
                "image": {
                    "encoding": "jpeg",
                    "data_base64": base64.b64encode(f"jpeg-{frame_id}".encode()).decode("ascii"),
                },
            },
            raw_size=10,
        )

    assert not (tmp_path / "frames" / "frame_000000.jpg").exists()
    assert (tmp_path / "frames" / "frame_000001.jpg").exists()
    assert (tmp_path / "frames" / "frame_000002.jpg").exists()
    assert (tmp_path / "latest.jpg").read_bytes() == b"jpeg-2"
