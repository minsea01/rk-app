#!/usr/bin/env python3
"""Generate a midterm-report figure pack from current repo artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path(".")
VIS_ROOT = ROOT / "artifacts" / "visualizations"
OUT_DIR = VIS_ROOT / "midterm_report_20260421"
CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")


TRAINING_COMPARE = {
    "COCO only": {"mAP50": 0.757, "Recall": 0.64, "F1": 0.72},
    "CrowdHuman+COCO": {"mAP50": 0.9044, "Recall": 0.95, "F1": 0.85},
}

ACHIEVEMENT_METRICS = {
    "训练侧 mAP@0.5": {"value": 90.44, "target": 90.0, "unit": "%", "mode": "ge"},
    "量化模型体积": {"value": 4.3, "target": 5.0, "unit": "MB", "mode": "le"},
    "单路实时性能": {"value": 33.08, "target": 30.0, "unit": "FPS", "mode": "ge"},
    "端到端延时": {"value": 30.23, "target": 45.0, "unit": "ms", "mode": "le"},
    "双网口吞吐": {"value": 912.0, "target": 900.0, "unit": "Mbps", "mode": "ge"},
}

MODEL_SIZE_MB = {
    "640 INT8": 4.6,
    "416 INT8": 4.3,
    "416 FP": 7.2,
}

MODEL_LATENCY_MS = {
    "640 INT8": 44.138,
    "416 INT8": 30.231,
    "416 FP": 58.965,
}

MODEL_VALID = {
    "640 INT8": False,
    "416 INT8": True,
    "416 FP": True,
}

MULTICORE_THROUGHPUT = 128.418

VIDEO_RUNTIME = {
    "15299460\n稀疏场景": {"fps": 17.2162, "latency": 59.325, "frames": 637, "avg_count": 2.8556},
    "14737069\n密集人群": {"fps": 17.5233, "latency": 56.817, "frames": 1507, "avg_count": 19.8640},
    "7624589\n中密度场景": {"fps": 25.1765, "latency": 39.558, "frames": 428, "avg_count": 4.1028},
}

TRACKER_IMPROVEMENT = {
    "before": {
        "max_count": 8.0,
        "avg_count": 4.1028,
        "avg_abs_diff": 0.2857,
        "stddev": 0.9539,
        "zero_frames": 0.0,
    },
    "after": {
        "max_count": 5.0,
        "avg_count": 3.5911,
        "avg_abs_diff": 0.0726,
        "stddev": 0.6928,
        "zero_frames": 1.0,
    },
}

STATUS_ITEMS = [
    ("RK3588 SSH/构建链", "已打通", "#2a9d8f"),
    ("RKNN / NPU 运行时", "已验证可用", "#2a9d8f"),
    ("detect_cli 板端编译", "已成功", "#2a9d8f"),
    ("folder -> RKNN -> TCP -> PC", "已闭环", "#2a9d8f"),
    ("GStreamer / aravissrc", "已补齐", "#2a9d8f"),
    ("416 INT8 主线", "已切换", "#2a9d8f"),
    ("三核并行推理", "128.4 FPS", "#5a189a"),
    ("GigE 真机相机", "待相机上电", "#e76f51"),
]

PEXELS_REPRESENTATIVES = [
    "pexels-12196195.jpg",
    "pexels-11891892.jpg",
    "pexels-20862422.jpg",
    "pexels-37144711.jpg",
]


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
        "DejaVu Sans",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    sans_fonts = [name for name in preferred_fonts if name in available_fonts]
    if "DejaVu Sans" not in sans_fonts:
        sans_fonts.append("DejaVu Sans")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": sans_fonts,
            "axes.unicode_minus": False,
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _annotate_bars(ax, bars, fmt="{:.1f}", dy=0.0, color="#1f1f1f") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight="bold",
            fontproperties=_font(10, "bold"),
        )


def _save(fig: plt.Figure, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / filename
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
        alpha=0.97,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.04 * w,
        y + h * 0.68,
        title,
        ha="left",
        va="center",
        fontsize=10,
        color="white",
        fontproperties=_font(10),
    )
    ax.text(
        x + 0.04 * w,
        y + h * 0.33,
        value,
        ha="left",
        va="center",
        fontsize=17,
        color="white",
        fontweight="bold",
        fontproperties=_font(17, "bold"),
    )


def fig_01_training_compare() -> Path:
    labels = ["mAP@0.5", "Recall", "F1"]
    coco = [TRAINING_COMPARE["COCO only"]["mAP50"], TRAINING_COMPARE["COCO only"]["Recall"], TRAINING_COMPARE["COCO only"]["F1"]]
    merged = [
        TRAINING_COMPARE["CrowdHuman+COCO"]["mAP50"],
        TRAINING_COMPARE["CrowdHuman+COCO"]["Recall"],
        TRAINING_COMPARE["CrowdHuman+COCO"]["F1"],
    ]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    bars1 = ax.bar(x - width / 2, coco, width, label="COCO only", color="#457b9d")
    bars2 = ax.bar(x + width / 2, merged, width, label="CrowdHuman + COCO", color="#2a9d8f")
    ax.set_title("阶段二训练侧精度提升对比", fontproperties=_font(16, "bold"))
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=_font(11))
    _annotate_bars(ax, bars1, fmt="{:.3f}", dy=0.02)
    _annotate_bars(ax, bars2, fmt="{:.3f}", dy=0.02)
    ax.legend(prop=_font(11))
    ax.text(
        0.5,
        -0.18,
        "说明：该图使用阶段二对比实验汇总值，体现 CrowdHuman 数据对密集人群和遮挡场景的补益。",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontproperties=_font(10),
    )
    fig.tight_layout()
    return _save(fig, "01_training_metric_comparison.png")


def fig_02_target_achievement() -> Path:
    labels = list(ACHIEVEMENT_METRICS.keys())
    values = [v["value"] for v in ACHIEVEMENT_METRICS.values()]
    targets = [v["target"] for v in ACHIEVEMENT_METRICS.values()]
    modes = [v["mode"] for v in ACHIEVEMENT_METRICS.values()]
    units = [v["unit"] for v in ACHIEVEMENT_METRICS.values()]

    normalized = []
    colors = []
    for value, target, mode in zip(values, targets, modes):
        if mode == "ge":
            ratio = value / target
            colors.append("#2a9d8f" if value >= target else "#e76f51")
        else:
            ratio = target / value
            colors.append("#2a9d8f" if value <= target else "#e76f51")
        normalized.append(ratio * 100.0)

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    bars = ax.bar(labels, normalized, color=colors, width=0.62)
    ax.axhline(100.0, color="#264653", linestyle="--", linewidth=1.4)
    ax.set_ylabel("相对任务指标达成度 (%)")
    ax.set_ylim(0, max(150.0, max(normalized) + 20.0))
    ax.set_title("任务指标达成情况", fontproperties=_font(16, "bold"))
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 3.0,
            f"{values[idx]:.2f} {units[idx]}\n目标 {targets[idx]:.2f} {units[idx]}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontproperties=_font(9),
        )
    fig.tight_layout()
    return _save(fig, "02_target_achievement_dashboard.png")


def fig_03_model_and_runtime() -> Path:
    labels = list(MODEL_SIZE_MB.keys())
    sizes = [MODEL_SIZE_MB[k] for k in labels]
    latency = [MODEL_LATENCY_MS[k] for k in labels]
    fps = [1000.0 / MODEL_LATENCY_MS[k] for k in labels]
    colors = ["#c1121f" if not MODEL_VALID[k] else "#2a9d8f" for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))

    bars0 = axes[0].bar(labels, sizes, color=["#3f7cac", "#2a9d8f", "#f4a261"], width=0.6)
    axes[0].axhline(5.0, color="#c1121f", linestyle="--", linewidth=1.3)
    axes[0].set_title("模型体积", fontproperties=_font(14, "bold"))
    axes[0].set_ylabel("MB")
    axes[0].set_ylim(0, 8.5)
    _annotate_bars(axes[0], bars0, fmt="{:.1f}", dy=0.1)

    bars1 = axes[1].bar(labels, latency, color=colors, width=0.6)
    axes[1].axhline(33.3, color="#264653", linestyle="--", linewidth=1.3)
    axes[1].set_title("单路端到端延时", fontproperties=_font(14, "bold"))
    axes[1].set_ylabel("ms")
    axes[1].set_ylim(0, 70)
    _annotate_bars(axes[1], bars1, fmt="{:.1f}", dy=1.0)

    bars2 = axes[2].bar(labels, fps, color=colors, width=0.6)
    axes[2].axhline(30.0, color="#264653", linestyle="--", linewidth=1.3)
    axes[2].set_title("单路 FPS", fontproperties=_font(14, "bold"))
    axes[2].set_ylabel("FPS")
    axes[2].set_ylim(0, 40)
    _annotate_bars(axes[2], bars2, fmt="{:.1f}", dy=0.5)
    axes[2].text(0, fps[0] + 2.0, "旧量化产物\n不可用", ha="center", color="#c1121f", fontproperties=_font(9))

    fig.suptitle("板端部署模型对比", fontsize=17, fontweight="bold", fontproperties=_font(17, "bold"))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "03_model_size_and_runtime_comparison.png")


def fig_04_multicore_and_status() -> Path:
    fig = plt.figure(figsize=(13.5, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(0.05, 0.94, "板端部署链路与三核并行成果", fontsize=20, fontweight="bold", fontproperties=_font(20, "bold"))
    _card(ax, 0.05, 0.80, 0.28, 0.10, "三核并行总吞吐", f"{MULTICORE_THROUGHPUT:.1f} FPS", "#5a189a")
    _card(ax, 0.36, 0.80, 0.28, 0.10, "单路有效模型", "416 INT8", "#2a9d8f")
    _card(ax, 0.67, 0.80, 0.28, 0.10, "结果回传链", "已验证", "#15616d")

    ax_bar = fig.add_axes([0.08, 0.48, 0.36, 0.24])
    bars = ax_bar.bar(["3-core throughput"], [MULTICORE_THROUGHPUT], color="#5a189a", width=0.45)
    ax_bar.axhline(30.0, color="#264653", linestyle="--", linewidth=1.2)
    ax_bar.set_ylim(0, 150)
    ax_bar.set_ylabel("FPS")
    ax_bar.set_title("detect_rknn_multicore 实测", fontproperties=_font(13, "bold"))
    _annotate_bars(ax_bar, bars, fmt="{:.1f}", dy=2.0)

    start_x = 0.50
    start_y = 0.68
    for idx, (title, value, color) in enumerate(STATUS_ITEMS):
        row = idx // 2
        col = idx % 2
        x = start_x + col * 0.22
        y = start_y - row * 0.14
        patch = FancyBboxPatch(
            (x, y),
            0.18,
            0.10,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0,
            facecolor=color,
            alpha=0.94,
        )
        ax.add_patch(patch)
        ax.text(x + 0.012, y + 0.062, title, color="white", fontsize=9, fontproperties=_font(9))
        ax.text(x + 0.012, y + 0.026, value, color="white", fontsize=13, fontweight="bold", fontproperties=_font(13, "bold"))

    fig.text(
        0.08,
        0.18,
        "说明：当前软件链已完成板端构建、NPU 推理、GStreamer/GigE 运行时补齐，以及 folder -> RKNN -> TCP -> PC 的闭环验证；\n唯一仍依赖现场硬件的环节是工业相机上电后的 GigE 真机采集闭环。",
        fontsize=11,
        fontproperties=_font(11),
    )
    return _save(fig, "04_multicore_and_deployment_status.png")


def fig_05_detection_result_compare() -> Path:
    panels = [
        ("原始图像", ROOT / "assets" / "test.jpg"),
        ("416 FP", VIS_ROOT / "board_detect_results" / "416_fp.jpg"),
        ("416 INT8", VIS_ROOT / "board_detect_results" / "416_norm_int8.jpg"),
        ("640 INT8（旧）", VIS_ROOT / "board_detect_results" / "640_int8.jpg"),
    ]
    captions = ["当前主线结果", "FP 基线", "板端实时主线", "旧量化产物对照"]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    fig.suptitle("板端模型检测结果对照", fontsize=18, fontweight="bold", fontproperties=_font(18, "bold"))
    for ax, (title, path), caption in zip(axes.flat, panels, captions):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(title, fontproperties=_font(14, "bold"))
        ax.text(
            0.02,
            0.04,
            caption,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=10,
            fontproperties=_font(10),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=(0, 0, 0, 0.65), edgecolor="none"),
        )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return _save(fig, "05_board_detection_result_comparison.png")


def fig_06_video_summary() -> Path:
    video_summaries = {
        "15299460\n稀疏场景": _read_json(VIS_ROOT / "video_check_15299460" / "summary.json"),
        "14737069\n密集人群": _read_json(VIS_ROOT / "video_check_14737069" / "summary.json"),
        "7624589\n中密度场景": _read_json(VIS_ROOT / "video_check_7624589" / "summary.json"),
    }
    labels = list(video_summaries.keys())
    avg_counts = [video_summaries[k]["avg_count"] for k in labels]
    zero_counts = [len(video_summaries[k]["zero_count_frames"]) for k in labels]
    fps = [VIDEO_RUNTIME[k]["fps"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
    bars0 = axes[0].bar(labels, avg_counts, color=["#457b9d", "#2a9d8f", "#f4a261"], width=0.62)
    axes[0].set_title("平均每帧人数", fontproperties=_font(14, "bold"))
    axes[0].set_ylabel("detections / frame")
    _annotate_bars(axes[0], bars0, fmt="{:.1f}", dy=0.2)

    bars1 = axes[1].bar(labels, zero_counts, color=["#c1121f", "#2a9d8f", "#2a9d8f"], width=0.62)
    axes[1].set_title("零检测帧数量", fontproperties=_font(14, "bold"))
    axes[1].set_ylabel("frames")
    _annotate_bars(axes[1], bars1, fmt="{:.0f}", dy=1.0)

    bars2 = axes[2].bar(labels, fps, color=["#457b9d", "#2a9d8f", "#f4a261"], width=0.62)
    axes[2].set_title("板端平均处理速度", fontproperties=_font(14, "bold"))
    axes[2].set_ylabel("FPS")
    _annotate_bars(axes[2], bars2, fmt="{:.1f}", dy=0.3)

    fig.suptitle("三段真实视频验证结果", fontsize=18, fontweight="bold", fontproperties=_font(18, "bold"))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, "06_video_runtime_summary.png")


def fig_07_video_case_montage() -> Path:
    paths = [
        ("稀疏场景样例", VIS_ROOT / "video_check_15299460" / "sample_montage.png"),
        ("密集人群样例", VIS_ROOT / "video_check_14737069" / "sample_montage.png"),
        ("中密度样例", VIS_ROOT / "video_check_7624589" / "sample_montage.png"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13.2, 14.0))
    fig.suptitle("真实视频检测样例拼图", fontsize=18, fontweight="bold", fontproperties=_font(18, "bold"))
    for ax, (title, path) in zip(axes, paths):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(title, fontproperties=_font(14, "bold"))
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    return _save(fig, "07_video_case_montage.png")


def fig_08_tracker_improvement() -> Path:
    metrics = ["max_count", "avg_count", "avg_abs_diff", "stddev", "zero_frames"]
    labels = ["最大人数", "平均人数", "帧间平均跳变", "人数标准差", "零检测帧"]
    before = [TRACKER_IMPROVEMENT["before"][k] for k in metrics]
    after = [TRACKER_IMPROVEMENT["after"][k] for k in metrics]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    bars1 = ax.bar(x - width / 2, before, width, label="调优前", color="#c1121f")
    bars2 = ax.bar(x + width / 2, after, width, label="调优后", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=_font(10))
    ax.set_title("轻量时序稳定化前后对比（7624589 视频）", fontproperties=_font(16, "bold"))
    ax.set_ylabel("指标值")
    _annotate_bars(ax, bars1, fmt="{:.2f}", dy=0.08)
    _annotate_bars(ax, bars2, fmt="{:.2f}", dy=0.08)
    ax.legend(prop=_font(11))
    ax.text(
        0.5,
        -0.16,
        "结论：引入 BoxTracker 后，重复框峰值与帧间人数跳变明显下降，视频观感更稳定。",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontproperties=_font(10),
    )
    fig.tight_layout()
    return _save(fig, "08_tracker_stabilization_improvement.png")


def fig_09_pexels_summary() -> Path:
    rows = _read_csv_rows(VIS_ROOT / "pexels_random15_20260420" / "summary.csv")
    rows_sorted = sorted(rows, key=lambda row: int(row["detections"]), reverse=True)
    labels = [row["filename"].replace(".jpg", "") for row in rows_sorted]
    counts = [int(row["detections"]) for row in rows_sorted]

    fig = plt.figure(figsize=(15.0, 12.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.18, wspace=0.18)

    ax_bar = fig.add_subplot(grid[:, 0])
    bars = ax_bar.barh(labels, counts, color="#3f7cac")
    ax_bar.invert_yaxis()
    ax_bar.set_title("Pexels 随机 15 张图片检测结果", fontproperties=_font(15, "bold"))
    ax_bar.set_xlabel("检测人数")
    for bar, value in zip(bars, counts):
        ax_bar.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2.0,
                    f"{value}", va="center", fontsize=9, fontproperties=_font(9))

    pair_titles = [
        "正常样例：3 -> 3",
        "密集合影：明显少检",
        "近景裁切：直接漏检",
        "暗光背身：多人被并框",
    ]
    right = grid[:, 1].subgridspec(4, 2, hspace=0.10, wspace=0.03)
    for idx, (filename, title) in enumerate(zip(PEXELS_REPRESENTATIVES, pair_titles)):
        ax_l = fig.add_subplot(right[idx, 0])
        ax_r = fig.add_subplot(right[idx, 1])
        ax_l.imshow(mpimg.imread(VIS_ROOT / "pexels_random15_20260420" / "input" / filename))
        ax_r.imshow(mpimg.imread(VIS_ROOT / "pexels_random15_20260420" / "output" / filename))
        ax_l.axis("off")
        ax_r.axis("off")
        ax_l.set_title(title, fontsize=10, fontproperties=_font(10, "bold"))
        ax_r.set_title("检测结果", fontsize=10, fontproperties=_font(10))

    fig.suptitle("泛化抽检：Pexels 图片与模型偏差", fontsize=18, fontweight="bold", fontproperties=_font(18, "bold"))
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    return _save(fig, "09_pexels_random15_generalization.png")


def fig_10_pack_overview() -> Path:
    figures = [
        "01_training_metric_comparison.png",
        "02_target_achievement_dashboard.png",
        "03_model_size_and_runtime_comparison.png",
        "04_multicore_and_deployment_status.png",
        "05_board_detection_result_comparison.png",
        "06_video_runtime_summary.png",
        "07_video_case_montage.png",
        "08_tracker_stabilization_improvement.png",
        "09_pexels_random15_generalization.png",
    ]
    titles = [
        "训练指标提升",
        "任务指标达成",
        "模型体积与实时性能",
        "板端部署与三核并行",
        "板端检测结果对照",
        "真实视频统计",
        "真实视频样例",
        "稳定化调优收益",
        "泛化抽检结果",
    ]

    fig = plt.figure(figsize=(14.2, 10.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.05, 0.95, "中期报告图包目录", fontsize=22, fontweight="bold", fontproperties=_font(22, "bold"))
    fig.text(0.05, 0.92, "以下 9 张图已按成果项整理，可直接插入报告正文。", fontsize=12, fontproperties=_font(12))

    start_y = 0.84
    for idx, (filename, title) in enumerate(zip(figures, titles), start=1):
        y = start_y - (idx - 1) * 0.085
        patch = FancyBboxPatch(
            (0.05, y - 0.035),
            0.90,
            0.055,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0,
            facecolor="#f1f5f9",
        )
        ax.add_patch(patch)
        fig.text(0.07, y, f"{idx:02d}. {title}", fontsize=13, fontweight="bold", fontproperties=_font(13, "bold"))
        fig.text(0.32, y, filename, fontsize=11, color="#475569", fontproperties=_font(11))

    fig.text(0.05, 0.10, f"输出目录：{OUT_DIR}", fontsize=11, fontproperties=_font(11))
    return _save(fig, "10_figure_pack_index.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        fig_01_training_compare(),
        fig_02_target_achievement(),
        fig_03_model_and_runtime(),
        fig_04_multicore_and_status(),
        fig_05_detection_result_compare(),
        fig_06_video_summary(),
        fig_07_video_case_montage(),
        fig_08_tracker_improvement(),
        fig_09_pexels_summary(),
        fig_10_pack_overview(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
