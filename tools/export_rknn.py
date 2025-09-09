#!/usr/bin/env python3
"""
RKNN Model Conversion Script
Converts ONNX YOLO model to RKNN format for RK3588 deployment
"""

import os
import sys
import argparse
from pathlib import Path

def convert_to_rknn(onnx_path, quant_dataset=None, output_name=None):
    """Convert ONNX model to RKNN format"""
    
    # Check if RKNN toolkit is available
    try:
        from rknn.api import RKNN
    except ImportError:
        print("❌ RKNN toolkit not found. Please install rknn-toolkit2")
        print("   pip install rknn-toolkit2")
        sys.exit(1)
    
    # Validate input
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        sys.exit(1)
    
    # Set output name
    if output_name is None:
        output_name = Path(onnx_path).stem + "_rk3588_int8.rknn"
    
    print("🔧 RKNN Conversion Configuration:")
    print(f"   - Input: {onnx_path}")
    print(f"   - Output: {output_name}")
    print(f"   - Target: RK3588")
    print(f"   - Quantization: {'INT8' if quant_dataset else 'FP16'}")
    if quant_dataset:
        print(f"   - Dataset: {quant_dataset}")
    
    # Create RKNN instance
    rknn = RKNN(verbose=False)
    
    try:
        # Configure RKNN
        print("\n📋 Configuring RKNN...")
        
        config_params = {
            'target_platform': 'rk3588',
            'optimization_level': 3,
            'output_optimize': True
        }
        
        if quant_dataset and os.path.exists(quant_dataset):
            # INT8 quantization
            config_params.update({
                'quantized_dtype': 'asymmetric_quantized-u8',
                'quantized_algorithm': 'normal',
                'quantized_method': 'channel',
                # Input preprocessing: 0-1 normalized -> u8 (0-255)
                'mean_values': [[0, 0, 0]],
                'std_values': [[255, 255, 255]],
                'reorder_channel': '0 1 2'  # RGB
            })
            print("   ✓ INT8 quantization enabled")
        else:
            # FP16 for development/testing
            config_params['quantized_dtype'] = 'asymmetric_quantized-u8'
            print("   ✓ Using FP16 precision")
        
        rknn.config(**config_params)
        
        # Load ONNX model
        print("📂 Loading ONNX model...")
        ret = rknn.load_onnx(onnx_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load ONNX model: {ret}")
        print("   ✓ ONNX model loaded")
        
        # Build RKNN model
        print("🔨 Building RKNN model...")
        build_params = {}
        if quant_dataset and os.path.exists(quant_dataset):
            build_params['do_quantization'] = True
            build_params['dataset'] = quant_dataset
        
        ret = rknn.build(**build_params)
        if ret != 0:
            raise RuntimeError(f"Failed to build RKNN model: {ret}")
        print("   ✓ RKNN model built")
        
        # Export RKNN model
        print("💾 Exporting RKNN model...")
        ret = rknn.export_rknn(output_name)
        if ret != 0:
            raise RuntimeError(f"Failed to export RKNN model: {ret}")
        
        print(f"✅ RKNN model exported: {output_name}")
        
        # Show file info
        if os.path.exists(output_name):
            file_size = os.path.getsize(output_name) / (1024 * 1024)  # MB
            print(f"📊 Model size: {file_size:.1f} MB")
            
        print("\n🚀 Deployment ready!")
        print("   Next steps:")
        print("   1. Copy model to RK3588 device")
        print("   2. Use RknnEngine in C++ application")
        print("   3. Test inference performance")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)
        
    finally:
        # Clean up
        rknn.release()

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX YOLO model to RKNN format")
    parser.add_argument("onnx_model", help="Path to ONNX model file")
    parser.add_argument("-d", "--dataset", help="Path to quantization dataset file")
    parser.add_argument("-o", "--output", help="Output RKNN model name")
    
    args = parser.parse_args()
    
    print("🎯 RKNN Model Conversion")
    print("=" * 40)
    
    convert_to_rknn(args.onnx_model, args.dataset, args.output)

if __name__ == "__main__":
    main()