from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def create_additional_charts_svg(filename):
    # Data 1: Pipeline Breakdown (Stacked Bar)
    # x86: ~137ms = Pre (15) + Infer (110) + Post (12)
    # RK3588 (Baseline): ~24ms = Pre (CPU 12) + Infer (NPU 10) + Post (2)
    # RK3588 (Opt): ~17.2ms = Pre (RGA 2.2) + Infer (NPU 15) + Post (0.1)

    stages = ["Preproc", "Inference", "Postproc", "Overhead"]
    colors = ["#f0a040", "#e05c5c", "#5b8dd9", "#888"]

    breakdown_data = [
        ("x86 CPU (ONNX)", [15, 110, 12, 0]),
        ("RK3588 (CPU Pre)", [12, 11, 2, 0.9]),
        ("RK3588 (RGA+ZeroCopy)", [2.2, 15, 0.1, 0.0]),
    ]

    # Data 2: Model Optimization
    model_data = [
        ("YOLOv8n FP32 (ONNX)", 12.8, 137.0),
        ("YOLOv8n INT8 (RKNN)", 4.9, 17.2),
        ("YOLOv8s FP32 (ONNX)", 44.8, 350.0),
        ("YOLOv8n-80map (RKNN)", 4.9, 17.5),
    ]

    # Canvas
    width = 1200
    height = 600
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
        'font-size="24" font-weight="bold">RK3588 优化成果详细分析</text>'
    )

    # --- Chart 1: Latency Breakdown (Stacked) ---
    x_offset = 50
    y_offset = 120
    chart_w = 500
    chart_h = 350

    svg += (
        f'<text x="{x_offset + chart_w / 2}" y="{y_offset - 20}" text-anchor="middle" '
        f'fill="{text_color}" font-size="18" font-weight="bold">管线耗时分解 (Stack Analysis)</text>'
    )

    svg += (
        f'<line x1="{x_offset}" y1="{y_offset}" x2="{x_offset}" y2="{y_offset + chart_h}" '
        f'stroke="{axis_color}" stroke-width="2"/>'
    )
    svg += (
        f'<line x1="{x_offset}" y1="{y_offset + chart_h}" x2="{x_offset + chart_w}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )

    bar_w = 60
    gap = 100
    max_val = 150

    leg_x = x_offset + chart_w - 150
    leg_y = y_offset + 20
    for i, stage in enumerate(stages):
        svg += (
            f'<rect x="{leg_x}" y="{leg_y + i * 25}" width="15" height="15" '
            f'fill="{colors[i]}" rx="2"/>'
        )
        svg += f'<text x="{leg_x + 25}" y="{leg_y + i * 25 + 12}" fill="#ccc" font-size="12">{stage}</text>'

    for i, (label, values) in enumerate(breakdown_data):
        cx = x_offset + 80 + i * (bar_w + gap)
        curr_y = y_offset + chart_h

        for j, val in enumerate(values):
            h = (val / max_val) * chart_h
            y = curr_y - h
            svg += (
                f'<rect x="{cx}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[j]}" '
                f'stroke="{bg_color}" stroke-width="1"/>'
            )
            if h > 15:
                svg += (
                    f'<text x="{cx + bar_w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" '
                    f'fill="white" font-size="10" font-weight="bold">{val}</text>'
                )
            curr_y = y

        svg += (
            f'<text x="{cx + bar_w / 2}" y="{y - 5}" text-anchor="middle" fill="{text_color}" '
            f'font-size="12" font-weight="bold">{sum(values):.1f}ms</text>'
        )

        parts = label.split(" ")
        for k, part in enumerate(parts):
            svg += (
                f'<text x="{cx + bar_w / 2}" y="{y_offset + chart_h + 20 + k * 15}" '
                f'text-anchor="middle" fill="#ccc" font-size="11">{part}</text>'
            )

    # --- Chart 2: Model Efficiency ---
    x_offset = 650
    y_offset = 120
    chart_w = 500

    svg += (
        f'<text x="{x_offset + chart_w / 2}" y="{y_offset - 20}" text-anchor="middle" '
        f'fill="{text_color}" font-size="18" font-weight="bold">模型轻量化与加速 (Size vs Latency)</text>'
    )

    svg += (
        f'<line x1="{x_offset}" y1="{y_offset}" x2="{x_offset}" y2="{y_offset + chart_h}" '
        f'stroke="{axis_color}" stroke-width="2"/>'
    )
    svg += (
        f'<line x1="{x_offset}" y1="{y_offset + chart_h}" x2="{x_offset + chart_w}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )
    svg += (
        f'<line x1="{x_offset + chart_w}" y1="{y_offset}" x2="{x_offset + chart_w}" '
        f'y2="{y_offset + chart_h}" stroke="{axis_color}" stroke-width="2"/>'
    )

    svg += (
        f'<text x="{x_offset - 30}" y="{y_offset + chart_h / 2}" '
        f'transform="rotate(-90, {x_offset - 30}, {y_offset + chart_h / 2})" '
        'text-anchor="middle" fill="#5b8dd9" font-size="14">模型体积 (MB)</text>'
    )
    svg += (
        f'<text x="{x_offset + chart_w + 30}" y="{y_offset + chart_h / 2}" '
        f'transform="rotate(90, {x_offset + chart_w + 30}, {y_offset + chart_h / 2})" '
        'text-anchor="middle" fill="#e05c5c" font-size="14">推理延迟 (ms)</text>'
    )

    max_size = 50
    max_lat = 400

    bar_w = 40
    gap = 80

    svg += f'<rect x="{x_offset + 20}" y="{y_offset + 20}" width="15" height="15" fill="#5b8dd9" rx="2"/>'
    svg += f'<text x="{x_offset + 45}" y="{y_offset + 32}" fill="#ccc" font-size="12">模型体积 (MB)</text>'
    svg += f'<rect x="{x_offset + 20}" y="{y_offset + 45}" width="15" height="15" fill="#e05c5c" rx="50%"/>'
    svg += f'<text x="{x_offset + 45}" y="{y_offset + 57}" fill="#ccc" font-size="12">推理延迟 (ms)</text>'

    points = []

    for i, (label, size, lat) in enumerate(model_data):
        cx = x_offset + 60 + i * (bar_w + gap)

        h = (size / max_size) * chart_h
        y = y_offset + chart_h - h
        svg += (
            f'<rect x="{cx - bar_w / 2}" y="{y}" width="{bar_w}" height="{h}" '
            'fill="#5b8dd9" opacity="0.7" rx="2"/>'
        )
        svg += (
            f'<text x="{cx}" y="{y - 5}" text-anchor="middle" fill="#5b8dd9" '
            f'font-size="12" font-weight="bold">{size}MB</text>'
        )

        py = y_offset + chart_h - (lat / max_lat) * chart_h
        points.append((cx, py))

        parts = label.split(" ")
        for k, part in enumerate(parts):
            svg += (
                f'<text x="{cx}" y="{y_offset + chart_h + 20 + k * 15}" '
                f'text-anchor="middle" fill="#ccc" font-size="10">{part}</text>'
            )

    svg += f'<polyline points="{" ".join([f"{x},{y}" for x, y in points])}" fill="none" stroke="#e05c5c" stroke-width="2"/>'

    for i, (x, y) in enumerate(points):
        svg += f'<circle cx="{x}" cy="{y}" r="5" fill="#e05c5c" stroke="white" stroke-width="2"/>'
        svg += (
            f'<text x="{x + 8}" y="{y - 8}" fill="#e05c5c" font-size="11" '
            f'font-weight="bold">{model_data[i][2]:.0f}ms</text>'
        )

    svg += "</svg>"

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(svg)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    create_additional_charts_svg(REPO_ROOT / "artifacts/optimization_analysis.svg")
