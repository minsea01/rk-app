import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def create_code_charts_svg(filename):
    # Data 1: Module Code Volume
    # Label, Lines, Color
    module_data = [
        ("Capture (Video Source)", 2271, "#5b8dd9"),
        ("Examples (Demo/Test)", 1907, "#888"),
        ("Infer/RKNN (NPU)", 1862, "#4caf7d"),
        ("Common (DMA/Mem)", 1295, "#f0a040"),
        ("Infer/ONNX (CPU)", 1121, "#e05c5c"),
        ("Preprocess (RGA)", 925, "#a05cbad9"),
        ("Pipeline (Main)", 917, "#5b8dd9"),
        ("Output (TCP/RTSP)", 638, "#5b8dd9"),
        ("Postprocess (NMS)", 442, "#5b8dd9"),
    ]

    # Data 2: Hardware API Usage
    # Label, Count, Color
    hw_data = [
        ("DMA-BUF (Zero-Copy)", 150, "#f0a040"),
        ("MPP (Decode)", 58, "#e05c5c"),
        ("RGA (Preproc)", 41, "#a05cbad9"),
        ("RKNN (Inference)", 40, "#4caf7d"),
    ]

    # Canvas
    width = 1000
    height = 500
    bg_color = "#0f1117"
    text_color = "white"
    axis_color = "#444"

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'style="background-color:{bg_color}; font-family: sans-serif;">'
    )

    # Title
    svg += (
        f'<text x="{width / 2}" y="40" text-anchor="middle" fill="{text_color}" '
        'font-size="24" font-weight="bold">RK3588 工程代码深度分析</text>'
    )
    svg += (
        f'<text x="{width / 2}" y="70" text-anchor="middle" fill="#888" font-size="14">'
        "Total: 11,378 Lines | 50 Source Files</text>"
    )

    # --- Chart 1: Module Volume (Horizontal Bar) ---
    x_offset = 50
    y_offset = 120
    chart_w = 400
    chart_h = 320

    svg += (
        f'<text x="{x_offset + chart_w / 2}" y="{y_offset - 20}" text-anchor="middle" '
        f'fill="{text_color}" font-size="18" font-weight="bold">代码模块规模 (SLOC)</text>'
    )

    svg += (
        f'<line x1="{x_offset + 120}" y1="{y_offset}" x2="{x_offset + 120}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )

    max_lines = 2500
    bar_h = 25
    gap = 10

    for i, (label, val, color) in enumerate(module_data):
        y = y_offset + i * (bar_h + gap) + 10
        bar_w = (val / max_lines) * (chart_w - 120)

        svg += (
            f'<text x="{x_offset + 110}" y="{y + bar_h / 2 + 5}" text-anchor="end" '
            f'fill="#ccc" font-size="12">{label}</text>'
        )
        svg += (
            f'<rect x="{x_offset + 120}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'fill="{color}" rx="4"/>'
        )
        svg += (
            f'<text x="{x_offset + 120 + bar_w + 10}" y="{y + bar_h / 2 + 5}" '
            f'fill="white" font-size="12" font-weight="bold">{val}</text>'
        )

    # --- Chart 2: Hardware API Usage (Pie Chart) ---
    x_offset = 650
    y_offset = 150
    radius = 120
    cx = x_offset + 150
    cy = y_offset + 130

    svg += (
        f'<text x="{cx}" y="{y_offset - 50}" text-anchor="middle" fill="{text_color}" '
        'font-size="18" font-weight="bold">硬件加速 API 调用分布</text>'
    )

    total_hw = sum(x[1] for x in hw_data)
    start_angle = 0

    for label, val, color in hw_data:
        percentage = val / total_hw
        sweep_angle = percentage * 360
        end_angle = start_angle + sweep_angle

        x1 = cx + radius * math.cos(math.radians(start_angle))
        y1 = cy + radius * math.sin(math.radians(start_angle))
        x2 = cx + radius * math.cos(math.radians(end_angle))
        y2 = cy + radius * math.sin(math.radians(end_angle))

        large_arc_flag = 1 if sweep_angle > 180 else 0

        path_d = f"M {cx} {cy} L {x1} {y1} A {radius} {radius} 0 {large_arc_flag} 1 {x2} {y2} Z"
        svg += f'<path d="{path_d}" fill="{color}" stroke="{bg_color}" stroke-width="2"/>'

        mid_angle = start_angle + sweep_angle / 2
        label_r = radius + 30
        lx = cx + label_r * math.cos(math.radians(mid_angle))
        ly = cy + label_r * math.sin(math.radians(mid_angle))

        anchor = "middle"
        if lx > cx + 10:
            anchor = "start"
        elif lx < cx - 10:
            anchor = "end"

        svg += (
            f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" fill="{text_color}" '
            f'font-size="12">{label}</text>'
        )
        svg += (
            f'<text x="{lx}" y="{ly + 15}" text-anchor="{anchor}" fill="#ccc" '
            f'font-size="11" font-weight="bold">{val}次 ({percentage * 100:.1f}%)</text>'
        )

        start_angle = end_angle

    svg += (
        f'<text x="{cx}" y="{cy + radius + 80}" text-anchor="middle" fill="#888" '
        f'font-size="12">Total API Calls: {total_hw} (Direct Driver/Lib Calls)</text>'
    )
    svg += (
        f'<text x="{cx}" y="{cy + radius + 100}" text-anchor="middle" fill="#888" '
        'font-size="12">Preproc(RGA + MPP) vs Infer(RKNN) ~ 3:1</text>'
    )

    svg += "</svg>"

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(svg)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    create_code_charts_svg(REPO_ROOT / "artifacts/code_stats.svg")
