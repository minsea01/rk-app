from rknn.api import RKNN

ONNX = 'artifacts/models/best.onnx'
RKNN_OUT = 'artifacts/models/yolo_industrial_640_int8.rknn'
CALIB_LIST = '/home/minsea01/datasets/industrial/quant.txt'

def main():
    rknn = RKNN(verbose=False)
    # 部分版本支持 core_mask，可尝试启用多核
    cfg = dict(
        target_platform='rk3588',
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
    )
    try:
        cfg['core_mask'] = 'rk_npu_core_all'
    except Exception:
        pass
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
