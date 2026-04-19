from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def create_bar_chart_svg(filename):
    # Data
    latency_data = [
        ("x86 CPU (ONNX)", 137.0, "#e05c5c"),
        ("RK3588 NPU (No RGA)", 23.9, "#f0a040"),
        ("RK3588 NPU (+RGA)", 17.2, "#4caf7d"),
    ]

    fps_data = [
        ("x86 CPU", 7.3, "#e05c5c"),
        ("RK3588 (No RGA)", 41.8, "#f0a040"),
        ("RK3588 (+RGA)", 58.2, "#4caf7d"),
    ]

    speedup_data = [
        ("NPU (No RGA)", 137.0 / 23.92, "#5b8dd9"),
        ("NPU (+RGA)", 137.0 / 17.17, "#7b6fd4"),
        ("NPU (+RGA+ZeroCopy)", 137.0 / 15.01, "#4caf7d"),
    ]

    # SVG Canvas
    width = 1200
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
        'font-size="24" font-weight="bold">RK3588 硬件加速推理性能对比</text>'
    )
    svg += (
        f'<text x="{width / 2}" y="70" text-anchor="middle" fill="#888" font-size="14">'
        "YOLOv8n INT8 640×640 | NPU 6 TOPS 3核</text>"
    )

    # Chart 1: Latency (Horizontal Bar)
    x_offset = 50
    y_offset = 120
    chart_w = 300
    chart_h = 300

    svg += (
        f'<text x="{x_offset + chart_w / 2}" y="{y_offset - 15}" text-anchor="middle" '
        f'fill="{text_color}" font-size="16" font-weight="bold">端到端推理延迟 (ms) ↓</text>'
    )

    svg += (
        f'<line x1="{x_offset}" y1="{y_offset}" x2="{x_offset}" y2="{y_offset + chart_h}" '
        f'stroke="{axis_color}" stroke-width="2"/>'
    )
    svg += (
        f'<line x1="{x_offset}" y1="{y_offset + chart_h}" x2="{x_offset + chart_w}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )

    max_val1 = 150
    bar_h = 40
    gap = 30

    for i, (label, val, color) in enumerate(latency_data):
        bar_w = (val / max_val1) * chart_w
        y = y_offset + i * (bar_h + gap) + 30

        svg += f'<rect x="{x_offset}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" rx="4"/>'
        svg += (
            f'<text x="{x_offset + bar_w + 10}" y="{y + bar_h / 2 + 5}" fill="{text_color}" '
            f'font-size="14" font-weight="bold">{val:.1f}</text>'
        )
        svg += f'<text x="{x_offset + 5}" y="{y - 8}" fill="#ccc" font-size="12">{label}</text>'

    # Chart 2: FPS (Vertical Bar)
    x_offset = 450
    y_offset = 120

    svg += (
        f'<text x="{x_offset + chart_w / 2}" y="{y_offset - 15}" text-anchor="middle" '
        f'fill="{text_color}" font-size="16" font-weight="bold">吞吐量 (FPS) ↑</text>'
    )

    svg += (
        f'<line x1="{x_offset}" y1="{y_offset}" x2="{x_offset}" y2="{y_offset + chart_h}" '
        f'stroke="{axis_color}" stroke-width="2"/>'
    )
    svg += (
        f'<line x1="{x_offset}" y1="{y_offset + chart_h}" x2="{x_offset + chart_w}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )

    target_y = y_offset + chart_h - (30 / 70) * chart_h
    svg += (
        f'<line x1="{x_offset}" y1="{target_y}" x2="{x_offset + chart_w}" y2="{target_y}" '
        'stroke="#888" stroke-width="1" stroke-dasharray="5,5"/>'
    )
    svg += (
        f'<text x="{x_offset + chart_w - 10}" y="{target_y - 5}" text-anchor="end" '
        'fill="#888" font-size="10">目标 30 FPS</text>'
    )

    max_val2 = 70
    bar_w = 50
    gap = 40
    start_x = x_offset + 30

    for i, (label, val, color) in enumerate(fps_data):
        bar_h_curr = (val / max_val2) * chart_h
        x = start_x + i * (bar_w + gap)
        y = y_offset + chart_h - bar_h_curr

        svg += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h_curr}" fill="{color}" rx="4"/>'
        svg += (
            f'<text x="{x + bar_w / 2}" y="{y - 10}" text-anchor="middle" fill="{text_color}" '
            f'font-size="14" font-weight="bold">{val:.1f}</text>'
        )
        parts = label.split(" ")
        for j, part in enumerate(parts):
            svg += (
                f'<text x="{x + bar_w / 2}" y="{y_offset + chart_h + 20 + j * 15}" '
                f'text-anchor="middle" fill="#ccc" font-size="11">{part}</text>'
            )

    # Chart 3: Speedup (Vertical Bar)
    x_offset = 850
    y_offset = 120

    svg += (
        f'<text x="{x_offset + chart_w / 2}" y="{y_offset - 15}" text-anchor="middle" '
        f'fill="{text_color}" font-size="16" font-weight="bold">相对 x86 加速比 (倍) ↑</text>'
    )

    svg += (
        f'<line x1="{x_offset}" y1="{y_offset}" x2="{x_offset}" y2="{y_offset + chart_h}" '
        f'stroke="{axis_color}" stroke-width="2"/>'
    )
    svg += (
        f'<line x1="{x_offset}" y1="{y_offset + chart_h}" x2="{x_offset + chart_w}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )

    max_val3 = 10

    for i, (label, val, color) in enumerate(speedup_data):
        bar_h_curr = (val / max_val3) * chart_h
        x = x_offset + 30 + i * (bar_w + gap)
        y = y_offset + chart_h - bar_h_curr

        svg += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h_curr}" fill="{color}" rx="4"/>'
        svg += (
            f'<text x="{x + bar_w / 2}" y="{y - 10}" text-anchor="middle" fill="{text_color}" '
            f'font-size="14" font-weight="bold">{val:.1f}x</text>'
        )
        parts = label.split(" ")
        for j, part in enumerate(parts):
            svg += (
                f'<text x="{x + bar_w / 2}" y="{y_offset + chart_h + 20 + j * 15}" '
                f'text-anchor="middle" fill="#ccc" font-size="11">{part}</text>'
            )

    svg += "</svg>"

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(svg)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    create_bar_chart_svg(REPO_ROOT / "artifacts/performance_chart.svg")
