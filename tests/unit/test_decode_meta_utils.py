#!/usr/bin/env python3
"""Unit tests for decode metadata helpers."""

import hashlib
import json
from pathlib import Path

import pytest

from apps.utils.decode_meta import (
    load_decode_meta,
    normalize_decode_meta,
    resolve_dfl_layout,
    resolve_head,
    resolve_raw_layout,
)


def test_normalize_decode_meta_alias_fields():
    meta = normalize_decode_meta(
        {
            "head": "DFL",
            "reg_max": "16",
            "classes": "80",
            "objectness": "false",
            "score_is_probability": "true",
            "coords_are_normalized": "true",
            "strides": [8, 16, 32],
        }
    )
    assert meta["head"] == "dfl"
    assert meta["reg_max"] == 16
    assert meta["num_classes"] == 80
    assert meta["has_objectness"] == 0
    assert meta["score_is_probability"] == 1
    assert meta["coords_are_normalized"] == 1
    assert meta["strides"] == (8, 16, 32)


def test_resolve_head_ambiguous_returns_none():
    assert resolve_head("auto", channels=84, decode_meta=None) is None


def test_resolve_layouts_with_metadata():
    decode_meta = {"head": "raw", "num_classes": 80, "has_objectness": 0}
    assert resolve_head("auto", channels=84, decode_meta=decode_meta) == "raw"
    assert resolve_raw_layout(84, decode_meta) == (False, 80)

    dfl_meta = {"head": "dfl", "reg_max": 16, "strides": [8, 16, 32], "num_classes": 80}
    assert resolve_head("auto", channels=144, decode_meta=dfl_meta) == "dfl"
    assert resolve_dfl_layout(144, dfl_meta) == (16, (8, 16, 32))


def test_load_decode_meta_from_model_sidecar(tmp_path):
    model_path = tmp_path / "demo.rknn"
    model_path.write_bytes(b"fake")

    sidecar = tmp_path / "demo.rknn.json"
    sidecar.write_text(
        json.dumps(
            {
                "head": "raw",
                "num_classes": 1,
                "has_objectness": 0,
                "score_is_probability": 1,
                "coords_are_normalized": 1,
            }
        )
    )

    meta = load_decode_meta(model_path)
    assert meta["head"] == "raw"
    assert meta["num_classes"] == 1
    assert meta["has_objectness"] == 0
    assert meta["score_is_probability"] == 1
    assert meta["coords_are_normalized"] == 1


def test_load_decode_meta_does_not_fallback_to_project_default(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.rknn"
    model_path.write_bytes(b"fake")

    project_default = tmp_path / "artifacts" / "models" / "decode_meta.json"
    project_default.parent.mkdir(parents=True)
    project_default.write_text(
        json.dumps(
            {
                "head": "dfl",
                "reg_max": 16,
                "num_classes": 80,
            }
        )
    )

    monkeypatch.chdir(tmp_path)
    meta = load_decode_meta(model_path)

    assert meta["head"] is None
    assert meta["reg_max"] is None
    assert meta["num_classes"] is None


def test_load_decode_meta_merges_adjacent_sidecars(tmp_path):
    model_path = tmp_path / "demo.rknn"
    model_path.write_bytes(b"fake")

    (tmp_path / "demo.rknn.json").write_text(json.dumps({"head": "raw"}))
    (tmp_path / "demo.rknn.meta").write_text("num_classes=3\nhas_objectness=false\n")

    meta = load_decode_meta(model_path)

    assert meta["head"] == "raw"
    assert meta["num_classes"] == 3
    assert meta["has_objectness"] == 0
    assert meta["score_is_probability"] is None
    assert meta["coords_are_normalized"] is None


def test_best_person_aug_int8_fixture_is_known_person_raw_model():
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "artifacts" / "models" / "best_person_aug_int8.rknn"
    if not model_path.exists():
        pytest.skip("deployment model binary is not available in this checkout")
    if not Path(f"{model_path}.json").exists() and not Path(f"{model_path}.meta").exists():
        pytest.skip("deployment model decode metadata sidecar is not available in this checkout")

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # This pins the tracked deployment fixture to the validated person model binary.
    # If the binary changes, the adjacent sidecar must be re-validated.
    assert digest == "afb0421fb79d7c83c2572598ba66066e1b363a0be58ddd065aa10c0495107283"

    meta = load_decode_meta(model_path)
    assert meta["head"] == "raw"
    assert meta["num_classes"] == 1
    assert meta["has_objectness"] == 0
    assert meta["score_is_probability"] == 1
    assert meta["coords_are_normalized"] is None
