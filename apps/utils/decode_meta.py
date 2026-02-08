"""Helpers for loading and applying decode metadata."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


DecodeMeta = Dict[str, Any]


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_head(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"dfl", "raw"}:
        return text
    return None


def _coerce_bool_flag(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return 1
    if text in {"0", "false"}:
        return 0
    return None


def _coerce_strides(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+", value)
        parsed = [int(x) for x in nums]
    else:
        try:
            parsed = [int(x) for x in value]
        except (TypeError, ValueError):
            return None
    parsed = [x for x in parsed if x > 0]
    if not parsed:
        return None
    return tuple(parsed)


def normalize_decode_meta(meta: Optional[Mapping[str, Any]]) -> DecodeMeta:
    """Normalize decode metadata to canonical keys with validated values."""
    out: DecodeMeta = {
        "head": None,
        "reg_max": None,
        "strides": None,
        "num_classes": None,
        "has_objectness": None,
    }
    if not isinstance(meta, Mapping):
        return out

    out["head"] = _coerce_head(meta.get("head"))

    reg_max = _coerce_int(meta.get("reg_max"))
    if reg_max is not None and reg_max > 0:
        out["reg_max"] = reg_max

    num_classes = _coerce_int(meta.get("num_classes"))
    if num_classes is None:
        num_classes = _coerce_int(meta.get("classes"))
    if num_classes is None:
        num_classes = _coerce_int(meta.get("nc"))
    if num_classes is not None and num_classes > 0:
        out["num_classes"] = num_classes

    has_obj = _coerce_bool_flag(meta.get("has_objectness"))
    if has_obj is None:
        has_obj = _coerce_bool_flag(meta.get("objectness"))
    if has_obj is None:
        has_obj = _coerce_bool_flag(meta.get("has_obj"))
    if has_obj is not None:
        out["has_objectness"] = has_obj

    strides = _coerce_strides(meta.get("strides"))
    if strides is not None:
        out["strides"] = strides

    return out


def _has_any(meta: DecodeMeta) -> bool:
    return any(meta.get(k) is not None for k in ("head", "reg_max", "strides", "num_classes", "has_objectness"))


def _parse_text_meta(content: str) -> DecodeMeta:
    def find_str(key: str) -> Optional[str]:
        m = re.search(
            rf'(?:^|[^A-Za-z0-9_])"?{re.escape(key)}"?\s*[:=]\s*"?([A-Za-z_]+)"?',
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return m.group(1) if m else None

    def find_int(key: str) -> Optional[int]:
        m = re.search(
            rf'(?:^|[^A-Za-z0-9_])"?{re.escape(key)}"?\s*[:=]\s*(-?\d+)',
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not m:
            return None
        return _coerce_int(m.group(1))

    def find_bool(key: str) -> Optional[int]:
        m = re.search(
            rf'(?:^|[^A-Za-z0-9_])"?{re.escape(key)}"?\s*[:=]\s*(true|false|0|1)',
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not m:
            return None
        return _coerce_bool_flag(m.group(1))

    def find_strides() -> Optional[Tuple[int, ...]]:
        m = re.search(
            r'(?:^|[^A-Za-z0-9_])"?strides"?\s*[:=]\s*\[([^\]]+)\]',
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not m:
            return None
        return _coerce_strides(m.group(1))

    raw: Dict[str, Any] = {}
    head = find_str("head")
    if head is not None:
        raw["head"] = head

    reg_max = find_int("reg_max")
    if reg_max is not None:
        raw["reg_max"] = reg_max

    for key in ("num_classes", "classes", "nc"):
        val = find_int(key)
        if val is not None:
            raw["num_classes"] = val
            break

    for key in ("has_objectness", "objectness", "has_obj"):
        val = find_bool(key)
        if val is not None:
            raw["has_objectness"] = val
            break

    strides = find_strides()
    if strides is not None:
        raw["strides"] = strides

    return normalize_decode_meta(raw)


def load_decode_meta(model_path: Optional[Path], logger: Optional[logging.Logger] = None) -> DecodeMeta:
    """Load decode metadata from model sidecars or project-level fallback."""
    candidates = []
    if model_path is not None:
        candidates.append(Path(f"{model_path}.json"))
        candidates.append(Path(f"{model_path}.meta"))
    candidates.append(Path("artifacts/models/decode_meta.json"))

    merged = normalize_decode_meta(None)
    for path in candidates:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not content.strip():
            continue

        parsed = normalize_decode_meta(None)
        try:
            loaded = json.loads(content)
            if isinstance(loaded, Mapping):
                parsed = normalize_decode_meta(loaded)
            else:
                parsed = _parse_text_meta(content)
        except json.JSONDecodeError:
            parsed = _parse_text_meta(content)

        if _has_any(parsed):
            for key in merged:
                if merged[key] is None and parsed.get(key) is not None:
                    merged[key] = parsed[key]
            if logger is not None:
                logger.info("Loaded decode metadata from %s", path)
            break

    return merged


def resolve_head(requested_head: str, channels: int, decode_meta: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Resolve decode head for current tensor channels."""
    if requested_head in {"dfl", "raw"}:
        return requested_head

    meta = normalize_decode_meta(decode_meta)
    if meta["head"] in {"dfl", "raw"}:
        return meta["head"]

    dfl_candidate = False
    raw_candidate = False
    reg_max = meta["reg_max"]
    num_classes = meta["num_classes"]
    has_obj = meta["has_objectness"]

    if reg_max is not None:
        cls_ch = channels - 4 * reg_max
        dfl_candidate = cls_ch > 0 and (num_classes is None or cls_ch == num_classes)

    if has_obj is not None:
        cls_ch = channels - (5 if has_obj == 1 else 4)
        raw_candidate = cls_ch >= 0 and (num_classes is None or cls_ch == num_classes)
    elif num_classes is not None:
        raw_candidate = channels in {4 + num_classes, 5 + num_classes}
    else:
        # Keep auto fallback only for compact raw heads; larger heads are ambiguous.
        raw_candidate = 5 <= channels < 64

    if dfl_candidate == raw_candidate:
        return None
    return "dfl" if dfl_candidate else "raw"


def resolve_dfl_layout(channels: int, decode_meta: Optional[Mapping[str, Any]]) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """Resolve (reg_max, strides) for DFL decode."""
    meta = normalize_decode_meta(decode_meta)
    reg_max = meta["reg_max"]
    num_classes = meta["num_classes"]

    if reg_max is None and num_classes is not None:
        remain = channels - num_classes
        if remain > 0 and remain % 4 == 0:
            reg_max = remain // 4
    if reg_max is None:
        reg_max = 16
    if reg_max <= 0:
        return None

    cls_ch = channels - 4 * reg_max
    if cls_ch <= 0:
        return None
    if num_classes is not None and cls_ch != num_classes:
        return None

    strides = meta["strides"] if meta["strides"] is not None else (8, 16, 32)
    return reg_max, tuple(strides)


def resolve_raw_layout(channels: int, decode_meta: Optional[Mapping[str, Any]]) -> Optional[Tuple[bool, int]]:
    """Resolve (has_objectness, num_classes) for RAW decode."""
    meta = normalize_decode_meta(decode_meta)
    num_classes = meta["num_classes"]
    has_obj = meta["has_objectness"]

    if has_obj is None:
        if num_classes is not None:
            if channels == 5 + num_classes:
                has_obj = 1
            elif channels == 4 + num_classes:
                has_obj = 0
            else:
                return None
        else:
            # Legacy fallback for compact heads when metadata is not available.
            has_obj = 1 if channels >= 5 else None
    if has_obj is None:
        return None

    cls_offset = 5 if has_obj == 1 else 4
    if num_classes is None:
        num_classes = channels - cls_offset
    if num_classes < 0 or cls_offset + num_classes != channels:
        return None
    if num_classes == 0 and has_obj == 0:
        return None

    return has_obj == 1, num_classes
