#!/usr/bin/env python3
"""Generate Chinese report figures for today's board-side work."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(".")
VIS_DIR = ROOT / "artifacts" / "visualizations"
DET_DIR = VIS_DIR / "board_detect_results"
CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")


def _font(size: float | None = None, weight: str | None = None):
    kwargs = {}
    if CJK_FONT_PATH.exists():
        kwargs["fname"] = str(CJK_FONT_PATH)
    if size is not None:
        kwargs["size"] = size
    if weight is not None:
        kwargs["weight"] = weight
    return font_manager.FontProperties(**kwargs)


def _style() -> None:
    for font_path in [
        str(CJK_FONT_PATH),
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    preferred_fonts = [
        "AR PL UMing CN",
        "AR PL SungtiL GB",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    fallback_fonts = [name for name in preferred_fonts if name in available_fonts]
    sans_fonts = ["DejaVu Sans"]
    sans_fonts.extend(name for name in fallback_fonts if name != "DejaVu Sans")

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": sans_fonts,
            "axes.unicode_minus": False,
        }
    )


def _load_last_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    if not text:
        return {}
    return json.loads(text[-1])


def _detection_caption(path: Path) -> str:
    obj = _load_last_json(path)
    dets = obj.get("detections", []) if isinstance(obj, dict) else []
    if not dets:
        return "0 个检测框"
    best = max(float(d.get("confidence", 0.0)) for d in dets)
    return f"{len(dets)} 个检测框\n最高置信度 {best:.3f}"


def generate_detection_comparison() -> Path:
    panels = [
        ("原始输入图像", ROOT / "assets" / "test.jpg", "原图"),
        (
            "416 INT8 板端结果",
            DET_DIR / "416_norm_int8.jpg",
            _detection_caption(DET_DIR / "best_person_aug_416_norm_int8.json"),
        ),
        (
            "416 FP 板端结果",
            DET_DIR / "416_fp.jpg",
            _detection_caption(DET_DIR / "best_person_aug_416_fp.json"),
        ),
        (
            "640 INT8 旧模型结果",
            DET_DIR / "640_int8.jpg",
            _detection_caption(DET_DIR / "best_person_aug_int8.json"),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    fig.suptitle(
        "RK3588 板端模型检测结果对比",
        fontsize=18,
        fontweight="bold",
        fontproperties=_font(18, "bold"),
    )

    for ax, (title, img_path, caption) in zip(axes.flat, panels):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontweight="bold", fontproperties=_font(14, "bold"))
        ax.axis("off")
        ax.text(
            0.02,
            0.04,
            caption,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="white",
            fontproperties=_font(10),
            bbox=dict(boxstyle="round,pad=0.35", facecolor=(0, 0, 0, 0.65), edgecolor="none"),
        )

    fig.text(
        0.5,
        0.02,
        "说明：前两张为今天修通后的有效板端结果；右下角 640 INT8 为旧量化产物，对照显示其当前仍为 0 检测。",
        ha="center",
        va="bottom",
        fontsize=10,
        fontproperties=_font(10),
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])

    out = VIS_DIR / "board_detection_result_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _card(ax, x: float, y: float, w: float, h: float, title: str, value: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=0,
        facecolor=color,
        alpha=0.96,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.03 * w,
        y + h * 0.70,
        title,
        ha="left",
        va="center",
        fontsize=11,
        color="white",
        fontproperties=_font(11),
    )
    ax.text(
        x + 0.03 * w,
        y + h * 0.33,
        value,
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="white",
        fontproperties=_font(18, "bold"),
    )


def generate_today_summary() -> Path:
    summary_img = mpimg.imread(VIS_DIR / "board_report_summary.png")
    det_img = mpimg.imread(VIS_DIR / "board_detection_result_comparison.png")

    fig = plt.figure(figsize=(14.0, 9.6))
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis("off")

    fig.text(
        0.05,
        0.955,
        "2026-04-20 当日实现与验证总览",
        fontsize=22,
        fontweight="bold",
        fontproperties=_font(22, "bold"),
    )
    fig.text(
        0.05,
        0.925,
        "聚焦 RK3588 板端 INT8 修通、端到端性能恢复、三核并行吞吐验证与真实检测结果导出。",
        fontsize=11,
        fontproperties=_font(11),
    )

    _card(ax_bg, 0.05, 0.79, 0.26, 0.10, "单路端到端延时", "30.23 ms", "#15616d")
    _card(ax_bg, 0.36, 0.79, 0.26, 0.10, "三核并行总吞吐", "128.4 FPS", "#5a189a")
    _card(ax_bg, 0.67, 0.79, 0.26, 0.10, "量化模型体积", "4.3 MB", "#2a9d8f")

    ax_l = fig.add_axes([0.05, 0.33, 0.43, 0.40])
    ax_l.imshow(summary_img)
    ax_l.set_title("性能与模型体积汇总图", fontweight="bold", fontproperties=_font(14, "bold"))
    ax_l.axis("off")

    ax_r = fig.add_axes([0.52, 0.19, 0.43, 0.54])
    ax_r.imshow(det_img)
    ax_r.set_title("板端真实检测结果对比", fontweight="bold", fontproperties=_font(14, "bold"))
    ax_r.axis("off")

    bullets = [
        "修复 C++ RKNN INT8 输入喂法：改为 UINT8 输入，由 runtime 负责量化转换。",
        "确认 416_norm_int8 在 detect_cli 板端链路中恢复正确检出，单张图 1 个目标。",
        "确认旧 640 INT8 量化产物当前仍不可用，可视化结果保留作对照证据。",
        "完成 detect_rknn_multicore 三引擎并行验证，3 核总吞吐达到 128.4 FPS。",
    ]
    fig.text(
        0.05, 0.24, "今日完成项", fontsize=15, fontweight="bold", fontproperties=_font(15, "bold")
    )
    y = 0.215
    for line in bullets:
        fig.text(0.055, y, "• " + line, fontsize=11, fontproperties=_font(11))
        y -= 0.035

    fig.text(
        0.05,
        0.06,
        "说明：单路数字来自 detect_cli；并行吞吐来自 detect_rknn_multicore（300 帧单图重复视频，3 个 RKNN engine 分别绑定 Core0/1/2）。",
        fontsize=10,
        fontproperties=_font(10),
    )

    out = VIS_DIR / "board_today_implementation_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        generate_detection_comparison(),
        generate_today_summary(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
