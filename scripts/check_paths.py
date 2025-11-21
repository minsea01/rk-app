#!/usr/bin/env python3
"""
路径完整性检查脚本
验证文件重组后所有路径引用是否正确
"""

import os
from pathlib import Path

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if Path(file_path).exists():
        print(f"[存在] {description}: {file_path}")
        return True
    else:
        print(f"[缺失] {description}: {file_path} - 文件不存在")
        return False

def main():
    print("RK3588项目路径完整性检查")
    print("=" * 50)
    
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"项目根目录: {project_root.absolute()}")
    print()
    
    issues = []
    
    # 检查核心模型文件
    print("模型文件检查:")
    models = [
        ("artifacts/models/best.onnx", "ONNX模型"),
        ("artifacts/models/industrial_15cls_rk3588_w8a8.rknn", "RKNN模型"),
        ("config/industrial_classes.txt", "15类工业标签")
    ]
    
    for path, desc in models:
        if not check_file_exists(path, desc):
            issues.append(f"缺失: {desc}")
    
    print()
    
    # 检查配置文件
    print("配置文件检查:")
    configs = [
        ("config/detection/detect.yaml", "检测配置"),
        ("config/deploy/rk3588_industrial_final.yaml", "RK3588部署配置"),
        ("config/detection/detect_rknn.yaml", "RKNN检测配置")
    ]
    
    for path, desc in configs:
        if not check_file_exists(path, desc):
            issues.append(f"缺失: {desc}")
    
    print()
    
    # 检查演示脚本
    print("演示脚本检查:")
    scripts = [
        ("scripts/demo/demo_presentation_script.sh", "主演示脚本"),
        ("achievement_report/现场演示脚本.sh", "成果演示脚本")
    ]
    
    for path, desc in scripts:
        if not check_file_exists(path, desc):
            issues.append(f"缺失: {desc}")
    
    print()
    
    # 检查日志文件
    print("测试数据检查:")
    logs = [
        ("logs/demo_results.log", "演示测试日志"),
    ]
    
    for path, desc in logs:
        if not check_file_exists(path, desc):
            issues.append(f"缺失: {desc}")
    
    print()
    
    # 检查技术文档
    print("技术文档检查:")
    docs = [
        ("docs/RK3588_VALIDATION_CHECKLIST.md", "硬件验证清单"),
        ("docs/guides/QUICK_START_GUIDE.md", "快速开始指南")
    ]
    
    for path, desc in docs:
        if not check_file_exists(path, desc):
            issues.append(f"缺失: {desc}")
    
    print()
    
    # 总结
    if not issues:
        print("所有文件路径检查通过！")
        print("[成功] 项目完整性验证成功")
        print()
        print("现在可以运行:")
        print("   ./scripts/demo/demo_presentation_script.sh    # 完整演示")
        print("   ./build/detect_cli --cfg config/detection/detect.yaml    # 快速测试")
        print("   cat logs/demo_results.log    # 查看测试数据")
    else:
        print("[警告] 发现以下问题:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("[建议] 检查文件位置或重新生成缺失文件")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
