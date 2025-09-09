#!/usr/bin/env python3
"""
RK3588 RKNN模型转换脚本
将YOLO ONNX模型转换为RK3588 NPU优化的RKNN格式
"""
from rknn.api import RKNN
import numpy as np
import cv2
from pathlib import Path
import os

def convert_onnx_to_rknn():
    """将ONNX模型转换为RKNN格式"""
    print("🔄 开始YOLO ONNX → RKNN转换（RK3588优化）")
    
    # 初始化RKNN
    rknn = RKNN(verbose=False)
    
    # RK3588平台配置
    print("⚙️ 配置RK3588目标平台...")
    rknn.config(
        target_platform='rk3588',
        quantized_dtype='asymmetric_quantized-u8',  # INT8量化
        optimization_level=3,                        # 最高优化级别
        output_optimize=1,                           # 输出优化
        mean_values=[[0, 0, 0]],                    # YOLO归一化
        std_values=[[255, 255, 255]],               # YOLO标准化
        reorder_channel='0 1 2'                     # RGB通道顺序
    )
    
    # 加载ONNX模型
    onnx_model_path = "../models/best.onnx"
    print(f"📥 加载ONNX模型: {onnx_model_path}")
    
    if not Path(onnx_model_path).exists():
        print(f"❌ ONNX模型不存在: {onnx_model_path}")
        print("请先复制best.onnx到models目录")
        return False
    
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print("❌ ONNX模型加载失败")
        return False
    
    # 准备量化数据集
    print("🎯 准备量化校准数据...")
    def load_calibration_dataset():
        """加载校准数据集"""
        dataset_path = Path("../models/calibration_images")
        if not dataset_path.exists():
            # 生成模拟校准数据
            print("📊 生成模拟校准数据...")
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(50):
                # 生成640x640的随机图像
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(dataset_path / f"calib_{i:03d}.jpg"), img)
        
        # 创建校准数据集列表文件
        calib_list = []
        for img_file in dataset_path.glob("*.jpg"):
            calib_list.append(str(img_file.absolute()))
        
        with open("../models/calibration_list.txt", "w") as f:
            for img_path in calib_list:
                f.write(f"{img_path}\n")
        
        return "../models/calibration_list.txt"
    
    # 构建RKNN模型
    print("🔧 构建RKNN模型（INT8量化 + NPU优化）...")
    dataset_file = load_calibration_dataset()
    
    ret = rknn.build(
        do_quantization=True,           # 启用INT8量化
        dataset=dataset_file,           # 量化校准数据集
        rknn_batch_size=1,             # 批处理大小
    )
    
    if ret != 0:
        print("❌ RKNN模型构建失败")
        return False
    
    # 导出RKNN模型
    rknn_output_path = "../models/yolo_industrial_rk3588.rknn"
    print(f"💾 导出RKNN模型: {rknn_output_path}")
    
    ret = rknn.export_rknn(rknn_output_path)
    if ret != 0:
        print("❌ RKNN模型导出失败")
        return False
    
    print("✅ RKNN转换成功完成！")
    
    # 性能评估
    print("📊 性能评估...")
    ret = rknn.eval_perf(inputs=[np.random.rand(1, 3, 640, 640).astype(np.float32)])
    
    # 精度分析
    print("🎯 精度分析...")
    try:
        ret = rknn.accuracy_analysis(
            inputs=[np.random.rand(1, 3, 640, 640).astype(np.float32)],
            output_dir="../models/accuracy_analysis"
        )
    except:
        print("⚠️ 精度分析跳过")
    
    rknn.release()
    
    print(f"\n🎉 RK3588 RKNN模型已准备就绪！")
    print(f"📁 模型路径: {rknn_output_path}")
    print("🚀 现在可以部署到RK3588开发板！")
    
    return True

if __name__ == "__main__":
    convert_onnx_to_rknn()
