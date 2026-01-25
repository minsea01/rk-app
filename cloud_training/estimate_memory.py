#!/usr/bin/env python3
"""
估算训练内存需求
用于确认 AutoDL 配置是否足够
"""

def estimate_memory():
    print("=" * 50)
    print("训练内存需求估算")
    print("=" * 50)

    # 数据集配置
    coco_images = 64000      # COCO Person
    crowdhuman_images = 15000  # CrowdHuman
    total_images = coco_images + crowdhuman_images

    # 图像配置
    img_size = 640
    channels = 3
    bytes_per_pixel = 1  # uint8

    # 单张图像大小 (MB)
    img_mb = (img_size * img_size * channels * bytes_per_pixel) / (1024 * 1024)

    print(f"\n数据集:")
    print(f"  COCO Person: {coco_images:,} 张")
    print(f"  CrowdHuman: {crowdhuman_images:,} 张")
    print(f"  总计: {total_images:,} 张")
    print(f"  单张图像: {img_mb:.2f} MB ({img_size}x{img_size})")

    # cache=ram 内存需求
    cache_ram_gb = (total_images * img_mb) / 1024
    print(f"\ncache=ram 需求:")
    print(f"  基础: {cache_ram_gb:.1f} GB")
    print(f"  + 预处理缓冲: ~{cache_ram_gb * 0.2:.1f} GB")
    print(f"  总计: ~{cache_ram_gb * 1.2:.1f} GB")

    # workers 内存
    workers = 16
    batch = 128
    mosaic_factor = 4  # mosaic 需要4张图

    workers_mem = workers * batch * mosaic_factor * img_mb / 1024
    print(f"\nworkers={workers} 峰值内存:")
    print(f"  batch={batch} × mosaic×4: ~{workers_mem:.1f} GB")

    # 总需求
    total_need = cache_ram_gb * 1.2 + workers_mem
    print(f"\n预估总需求: {total_need:.1f} GB")

    # 对比可用
    available = 90
    print(f"可用 RAM: {available} GB")

    if total_need > available:
        print(f"\n⚠️  可能不足! 差 {total_need - available:.1f} GB")
        print("\n建议:")
        print("  1. 减少 workers 到 8")
        print("  2. 或使用 cache=disk")
        print("  3. 或分开训练 (先COCO，再微调)")
    else:
        print(f"\n✅ 内存充足 (余量 {available - total_need:.1f} GB)")

    # GPU 显存估算
    print("\n" + "=" * 50)
    print("GPU 显存估算")
    print("=" * 50)

    # YOLOv8n 模型大小
    model_mb = 6  # YOLOv8n ~6MB 参数
    model_gpu_mb = model_mb * 4  # FP32

    # 激活值 (粗略估算)
    activation_mb = batch * img_size * img_size * 64 / (1024 * 1024) * 0.1  # 压缩系数

    # 梯度
    grad_mb = model_mb * 4

    # 优化器状态 (AdamW = 2x)
    optimizer_mb = model_mb * 4 * 2

    gpu_total = model_gpu_mb + activation_mb + grad_mb + optimizer_mb
    print(f"  模型: {model_gpu_mb:.0f} MB")
    print(f"  激活值 (batch={batch}): ~{activation_mb:.0f} MB")
    print(f"  梯度: {grad_mb:.0f} MB")
    print(f"  优化器: {optimizer_mb:.0f} MB")
    print(f"  预估总需求: ~{gpu_total/1024:.1f} GB")
    print(f"  可用: 26 GB")

    if gpu_total / 1024 > 26:
        print(f"  ⚠️  显存可能不足")
    else:
        print(f"  ✅ 显存充足")


if __name__ == "__main__":
    estimate_memory()
