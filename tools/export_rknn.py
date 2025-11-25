#!/usr/bin/env python3
"""RKNN Model Conversion Script.

Converts ONNX YOLO model to RKNN format for RK3588 deployment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def convert_to_rknn(
    onnx_path: str,
    quant_dataset: Optional[str] = None,
    output_name: Optional[str] = None
) -> None:
    """Convert ONNX model to RKNN format.
    
    Args:
        onnx_path: Path to input ONNX model
        quant_dataset: Path to quantization dataset file (optional)
        output_name: Output RKNN model filename (optional)
        
    Raises:
        ImportError: If RKNN toolkit is not installed
        FileNotFoundError: If ONNX model not found
        RuntimeError: If conversion fails
    """
    # Check if RKNN toolkit is available
    try:
        from rknn.api import RKNN
    except ImportError as e:
        logger.error("❌ RKNN toolkit not found. Please install rknn-toolkit2")
        logger.error("   pip install rknn-toolkit2")
        raise ImportError("rknn-toolkit2 not installed") from e
    
    # Validate input
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Set output name
    if output_name is None:
        output_name = Path(onnx_path).stem + "_rk3588_int8.rknn"
    
    logger.info("🔧 RKNN Conversion Configuration:")
    logger.info(f"   - Input: {onnx_path}")
    logger.info(f"   - Output: {output_name}")
    logger.info(f"   - Target: RK3588")
    logger.info(f"   - Quantization: {'INT8' if quant_dataset else 'FP16'}")
    if quant_dataset:
        logger.info(f"   - Dataset: {quant_dataset}")
    
    # Create RKNN instance
    rknn = RKNN(verbose=False)
    
    try:
        # Configure RKNN
        logger.info("\n📋 Configuring RKNN...")
        
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
            logger.info("   ✓ INT8 quantization enabled")
        else:
            # FP16 for development/testing
            config_params['quantized_dtype'] = 'asymmetric_quantized-u8'
            logger.info("   ✓ Using FP16 precision")
        
        rknn.config(**config_params)
        
        # Load ONNX model
        logger.info("📂 Loading ONNX model...")
        ret = rknn.load_onnx(onnx_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load ONNX model: {ret}")
        logger.info("   ✓ ONNX model loaded")
        
        # Build RKNN model
        logger.info("🔨 Building RKNN model...")
        build_params = {}
        if quant_dataset and os.path.exists(quant_dataset):
            build_params['do_quantization'] = True
            build_params['dataset'] = quant_dataset
        
        ret = rknn.build(**build_params)
        if ret != 0:
            raise RuntimeError(f"Failed to build RKNN model: {ret}")
        logger.info("   ✓ RKNN model built")
        
        # Export RKNN model
        logger.info("💾 Exporting RKNN model...")
        ret = rknn.export_rknn(output_name)
        if ret != 0:
            raise RuntimeError(f"Failed to export RKNN model: {ret}")
        
        logger.info(f"✅ RKNN model exported: {output_name}")
        
        # Show file info
        if os.path.exists(output_name):
            file_size = os.path.getsize(output_name) / (1024 * 1024)  # MB
            logger.info(f"📊 Model size: {file_size:.1f} MB")
            
        logger.info("\n🚀 Deployment ready!")
        logger.info("   Next steps:")
        logger.info("   1. Copy model to RK3588 device")
        logger.info("   2. Use RknnEngine in C++ application")
        logger.info("   3. Test inference performance")
        
    except RuntimeError:
        raise
        
    finally:
        # Clean up
        rknn.release()

def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(description="Convert ONNX YOLO model to RKNN format")
    parser.add_argument("onnx_model", help="Path to ONNX model file")
    parser.add_argument("-d", "--dataset", help="Path to quantization dataset file")
    parser.add_argument("-o", "--output", help="Output RKNN model name")
    
    args = parser.parse_args()
    
    logger.info("🎯 RKNN Model Conversion")
    logger.info("=" * 40)
    
    try:
        convert_to_rknn(args.onnx_model, args.dataset, args.output)
        return 0
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        logger.error(f"❌ Conversion failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())