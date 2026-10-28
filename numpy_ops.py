"""Optional single-thread BLAS prototype for CPUs where ORT integer GEMM is slow."""

import numpy as np
from onnxruntime_extensions import (
    onnx_op,
    PyCustomOpDef,
    get_library_path,
    enable_py_op,
)

enable_py_op(True)


@onnx_op(
    op_type="PackedW8MatMul",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def packed_w8_matmul(x, weight, scales):
    # Decode one layer at a time; never materialize the whole FP32 model in RAM.
    dense = weight.astype(np.float32)
    dense *= scales
    return np.matmul(x, dense).astype(np.float32, copy=False)


def library_path():
    return get_library_path()
