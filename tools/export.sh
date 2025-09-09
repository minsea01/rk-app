#!/bin/bash
# YOLO Model Export Script - Fixed Specification
# Exports ONNX model with consistent parameters for deployment

set -e

# Configuration
MODEL_PATH="${1:-best.pt}"
OUTPUT_NAME="best_static640"
IMG_SIZE=640
OPSET=12

echo "🚀 Exporting YOLO model: $MODEL_PATH"
echo "📋 Configuration:"
echo "   - Input size: ${IMG_SIZE}x${IMG_SIZE}"
echo "   - Format: ONNX (opset $OPSET)"
echo "   - Output: ${OUTPUT_NAME}.onnx"
echo "   - NMS: Disabled (for custom post-processing)"

# Activate YOLO training environment
source /home/minsea01/yolo-train/bin/activate

# Export model with fixed parameters
/home/minsea01/yolo-train/bin/python -c "
from ultralytics import YOLO

# Load model
model = YOLO('$MODEL_PATH')

# Export with fixed parameters
model.export(
    format='onnx',
    imgsz=$IMG_SIZE,
    opset=$OPSET,
    simplify=True,
    dynamic=False,  # Static shape for optimization
    half=False      # Keep FP32 for better compatibility
)

print('✅ ONNX export completed')
"

# Check if export was successful
if [ -f "${MODEL_PATH%.*}.onnx" ]; then
    # Rename to standard name
    mv "${MODEL_PATH%.*}.onnx" "${OUTPUT_NAME}.onnx"
    echo "✅ Model exported successfully: ${OUTPUT_NAME}.onnx"
    
    # Show model info
    echo ""
    echo "📊 Model Information:"
    ls -lh "${OUTPUT_NAME}.onnx"
    
    echo ""
    echo "🔧 Model Specifications:"
    echo "   - Input: RGB image, NCHW format, 0-1 normalized"
    echo "   - Shape: [1, 3, $IMG_SIZE, $IMG_SIZE]"
    echo "   - Output: Raw predictions without NMS"
    echo "   - Classes: 4 (person, bed, dining_table, laptop)"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Use this model for RKNN conversion"
    echo "   2. Test with C++ inference engine"
    echo "   3. Prepare quantization dataset"
    
else
    echo "❌ Export failed - ONNX file not found"
    exit 1
fi