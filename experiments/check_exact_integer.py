"""Check the native ORT operator against standard MatMulInteger on an AVX2 host."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # Set single-thread environment before numerical libraries.
import numpy as np
import onnxruntime as ort
import psutil
from onnx import TensorProto, helper, numpy_helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/round6_exact_abi_check.json")
    parser.add_argument("--library", help="Check a versioned candidate without replacing the active library")
    args = parser.parse_args()
    if hasattr(psutil.Process(), "cpu_affinity"):
        psutil.Process().cpu_affinity([0])
    rng = np.random.RandomState(619)
    rows = []
    for shape, k, n, zero, pattern in [
        ([1, 1], 12, 8, 0, "random64"),
        ([1, 5], 144, 512, 255, "random64"),
        ([2, 4], 512, 1536, 127, "random64"),
        ([1, 15], 1536, 512, 71, "random64"),
        ([1, 8], 8, 8, 0, "positive32_safe_boundary"),
        ([1, 8], 16, 16, 0, "positive64_requires_fallback"),
        ([1, 8], 8, 8, 255, "negative32_safe_boundary"),
        ([1, 17], 512, 16, 128, "random32_all_safe"),
        ([1, 3], 8, 16, 0, "positive32_safe_boundary"),
        ([1, 3], 144, 24, 127, "random64"),
        ([1, 9], 64, 32, 255, "sparse_unsafe_spans"),
        ([1, 4], 65536, 16, 0, "random32_all_safe"),
        ([1, 6], 16, 32, 127, "random64"),
        ([1, 2], 8, 16, 255, "negative32_safe_boundary"),
        ([1, 13], 16, 16, 0, "positive64_requires_fallback"),
        ([1, 12], 8, 16, 255, "negative32_safe_boundary"),
        ([1, 12], 8, 8, 0, "positive64_requires_fallback"),
    ]:
        x = rng.randint(0, 256, (*shape, k), dtype=np.uint8)
        limit = 32 if pattern == "random32_all_safe" else 64
        weights = rng.randint(-limit, limit + 1, (k, n)).astype(np.int8)
        if pattern.startswith(("positive", "negative")):
            value = -32 if pattern.startswith("negative") else 64 if "64" in pattern else 32
            weights.fill(value)
            x.fill(255)
        elif pattern == "sparse_unsafe_spans":
            weights.fill(0)
            # First/last unsafe blocks, adjacent blocks, and flags that come
            # from either of the two N8 panels merged into an N16 tile.
            for block, column, value in [(0, 2, 64), (2, 12, 64),
                                         (3, 2, -64), (7, 12, -64)]:
                weights[block * 8:block * 8 + 8, column] = value
            x.fill(255)
        initializers = [
            numpy_helper.from_array(weights, "w"),
            numpy_helper.from_array(np.array(zero, np.uint8), "az"),
            numpy_helper.from_array(np.zeros(n, np.int8), "bz"),
        ]
        outputs = []
        for custom in [False, True]:
            node = helper.make_node(
                "ExactMatMulInteger" if custom else "MatMulInteger",
                ["x", "w", "az", "bz"],
                ["y"],
                domain="com.supertonic.exact" if custom else "",
            )
            graph = helper.make_graph(
                [node], "integer_check",
                [helper.make_tensor_value_info("x", TensorProto.UINT8, [*shape, k])],
                [helper.make_tensor_value_info("y", TensorProto.INT32, [*shape, n])],
                initializers,
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.supertonic.exact", 1)],
                ir_version=10,
            )
            options = runtime.options(exact_int8_backend=custom and not args.library)
            if custom and args.library:
                options.enable_mem_pattern = False
                options.register_custom_ops_library(args.library)
            session = ort.InferenceSession(
                model.SerializeToString(), options,
                providers=["CPUExecutionProvider"],
            )
            first = session.run(None, {"x": x})[0]
            repeated = session.run(None, {"x": x})[0]
            np.testing.assert_array_equal(first, repeated)
            outputs.append(first)
            del session
        np.testing.assert_array_equal(*outputs)
        rows.append({
            "shape": [*shape, k, n], "zero_point": zero, "weight_pattern": pattern,
            "s32_bit_identical": True, "persistent_cache_repeat_identical": True,
        })
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
