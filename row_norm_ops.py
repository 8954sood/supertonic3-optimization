"""Optional ARM64 normalization, gated by exact checks against installed ORT."""
import ctypes
from functools import lru_cache
from pathlib import Path
import platform

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


@lru_cache(maxsize=8)
def library_path(override=None, *, widths=(512,)):
    if platform.machine().lower() not in {'arm64','aarch64'} or platform.system() not in {'Darwin','Linux'}:
        raise RuntimeError('The optional row norm kernel needs ARM64; use the standard ORT package')
    if not widths or any(width not in {64,256,512} for width in widths):
        raise ValueError('Supported row normalization widths are 64, 256 and 512')
    path=Path(override).resolve() if override else Path(__file__).resolve().parent/'native'/('librow_norm.dylib' if platform.system()=='Darwin' else 'librow_norm.so')
    if not path.is_file():
        raise FileNotFoundError('Build the optional kernel first: python native/build_row_norm.py')
    library=ctypes.CDLL(str(path))
    function=library.row_norm
    function.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64,ctypes.c_int64,ctypes.c_float]
    function.restype=None
    # Verify the installed ORT's numerical convention on the target CPU before
    # accepting the library. No cross-machine precomputed statistics are used.
    model=helper.make_model(helper.make_graph([
        helper.make_node('LayerNormalization',['x','scale','bias'],['y'],axis=-1,epsilon=1e-6)],
        'normalization_compatibility',[
            helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,'T','C']),
            helper.make_tensor_value_info('scale',TensorProto.FLOAT,['C']),
            helper.make_tensor_value_info('bias',TensorProto.FLOAT,['C'])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,'T','C'])]),
        opset_imports=[helper.make_opsetid('',17)],ir_version=10)
    settings=ort.SessionOptions();settings.intra_op_num_threads=1;settings.inter_op_num_threads=1
    session=ort.InferenceSession(model.SerializeToString(),settings,providers=['CPUExecutionProvider'])
    rng=np.random.RandomState(91)
    for width in widths:
        for rows in [1,4,5,8,9,16,17,32,33,81]:
            x=rng.randn(1,rows,width).astype(np.float32)
            scale=rng.randn(width).astype(np.float32);bias=rng.randn(width).astype(np.float32)
            expected=session.run(None,dict(x=x,scale=scale,bias=bias))[0]
            output=np.empty_like(x)
            function(x.ctypes.data,scale.ctypes.data,bias.ctypes.data,output.ctypes.data,rows,width,1e-6)
            if not np.array_equal(expected,output) or expected.tobytes()!=output.tobytes():
                raise RuntimeError('Row normalization differs from installed ORT; use the standard ORT package')
    return str(path)
