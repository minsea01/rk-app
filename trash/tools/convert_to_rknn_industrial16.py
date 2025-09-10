from rknn.api import RKNN
import os

ONNX = os.environ.get('IND16_ONNX', 'artifacts/models/best.onnx')
RKNN_OUT = os.environ.get('IND16_RKNN', 'artifacts/models/industrial_15cls_rk3588_w8a8.rknn')
CALIB_LIST = os.environ.get('IND16_CALIB', '/home/minsea01/datasets/industrial/quant.txt')

def main():
    rknn = RKNN(verbose=False)
    cfg = dict(
        target_platform='rk3588',
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]], 
        quantized_dtype='w8a8',  # RKNN 2.3.2新格式：权重8位+激活8位
        optimization_level=3,
        output_optimize=True,
    )
    rknn.config(**cfg)

    print('Loading ONNX:', ONNX)
    ret = rknn.load_onnx(model=ONNX, inputs=['images'], input_size_list=[[1,3,640,640]])
    if ret != 0:
        raise RuntimeError('load_onnx failed: %d' % ret)

    print('Building RKNN with quantization...')
    ret = rknn.build(do_quantization=True, dataset=CALIB_LIST, pre_compile=True)
    if ret != 0:
        raise RuntimeError('build failed: %d' % ret)

    print('Exporting RKNN to', RKNN_OUT)
    ret = rknn.export_rknn(RKNN_OUT)
    if ret != 0:
        raise RuntimeError('export failed: %d' % ret)
    print('OK ->', RKNN_OUT)

if __name__ == '__main__':
    main()

