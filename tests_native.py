"""Check native custom-op ABI, shapes and arithmetic on Linux before real TTS."""

from runtime import options
import numpy as np
from onnx import helper, TensorProto
import onnxruntime as ort

rng = np.random.RandomState(31)
for shape, n in [((2, 3), 7), ((1, 4, 5), 8), ((2, 3, 7), 9)]:
    k = shape[-1]
    x = rng.randn(*shape).astype(np.float32)
    w = rng.randint(-127, 128, (k, n)).astype(np.int8)
    s = rng.uniform(0.001, 0.1, n).astype(np.float32)
    node = helper.make_node(
        "PackedW8MatMul", ["x", "w", "s"], ["y"], domain="com.supertonic.blas"
    )
    graph = helper.make_graph(
        [node],
        "native_check",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape)),
            helper.make_tensor_value_info("w", TensorProto.INT8, [k, n]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [n]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape[:-1]) + [n])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.supertonic.blas", 1),
        ],
        ir_version=10,
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        options(native_backend=True),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {"x": x, "w": w, "s": s})[0]
    expected = x @ (w.astype(np.float32) * s)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("passed", shape, n, flush=True)
