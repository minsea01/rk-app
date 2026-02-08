#!/usr/bin/env python3
"""Unit tests for decode metadata helpers."""

import json

from apps.utils.decode_meta import (
    load_decode_meta,
    normalize_decode_meta,
    resolve_dfl_layout,
    resolve_head,
    resolve_raw_layout,
)


def test_normalize_decode_meta_alias_fields():
    meta = normalize_decode_meta({
        "head": "DFL",
        "reg_max": "16",
        "classes": "80",
        "objectness": "false",
        "strides": [8, 16, 32],
    })
    assert meta["head"] == "dfl"
    assert meta["reg_max"] == 16
    assert meta["num_classes"] == 80
    assert meta["has_objectness"] == 0
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
    sidecar.write_text(json.dumps({
        "head": "raw",
        "num_classes": 1,
        "has_objectness": 0,
    }))

    meta = load_decode_meta(model_path)
    assert meta["head"] == "raw"
    assert meta["num_classes"] == 1
    assert meta["has_objectness"] == 0
